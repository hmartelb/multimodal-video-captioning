import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.temporal_attention import TemporalAttention


class FeaturesCaptioning(nn.Module):
    def __init__(
        self,
        in_feature_size,
        output_size,  # vocab_size
        rnn_type="LSTM",
        rnn_num_layers=1,
        rnn_bidirectional=False,
        rnn_hidden_size=128,
        rnn_dropout=0.5,
        embedding_size=128, # vocab_embedding_size
        attn_size=128,
        device = 'cpu',
        **args,
    ):
        super(FeaturesCaptioning, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = rnn_num_layers
        self.num_directions = 2 if rnn_bidirectional else 1
        self.feature_size = in_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = rnn_hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.rnn_dropout_p = rnn_dropout if self.num_layers > 1 else 0
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)

        # if self.attn_size > 0:
        self.attention = TemporalAttention(
            hidden_size=self.num_directions * self.hidden_size, 
            feature_size=self.feature_size, 
            bottleneck_size=self.attn_size
        ) 
        # else:
        #     self.attention = nn.Identity() # FIXME: 

        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(
            input_size=self.embedding_size + self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False,
        )

        self.out = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def _last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), self.num_directions * last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden

    def _init_hidden(self, batch_size):
        if self.rnn_type == "LSTM":
            hidden = (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device),
            )
        else:
            hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden = hidden.to(self.device)
        return hidden

    def forward_word(self, features, hidden, previous_words):
        embedded = self.embedding(previous_words)

        last_hidden = self._last_hidden(hidden)
        features, attn_weights = self.attention(last_hidden, features)

        input_combined = torch.cat((embedded, features.unsqueeze(0)), dim=2)
        output, hidden = self.rnn(input_combined, hidden)

        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    
    def forward_sentence(self, features, captions, hidden, max_caption_len=30, teacher_forcing_ratio=1):
        batch_size = features.shape[0]  # Shape [batch_size, n_seconds, feature_size]

        # Placeholder variables initialization
        # (max_caption_len + 2 because we add 2 special tokens: <SOS> and <EOS>)
        sentence = Variable(torch.zeros(max_caption_len, batch_size, self.output_size)).to(self.device)
        D, B, H = (hidden[0] if self.rnn_type == "LSTM" else hidden).shape
        hidden_states = Variable(torch.zeros(max_caption_len, D, B, H)).to(self.device)
        output = Variable(torch.LongTensor(1, batch_size).fill_(1)).to(self.device)  # self.vocab.stoi["<SOS>"]

        for t in range(1, max_caption_len):
            # Get the next word
            output, hidden, attn_weights = self.forward_word(features, hidden, output.view(1, -1))

            # Update result and hidden state for next prediction
            sentence[t] = output
            hidden_states[t] = hidden[0] if self.rnn_type == "LSTM" else hidden

            top1 = output.data.max(1)[1]
            
            # Apply teacher forcing with probability, disable when caption is None
            # YES: take true word, NO: take prediction
            if captions is None:
                output = top1
            else:
                is_teacher = torch.rand(1) < teacher_forcing_ratio
                output = Variable(captions.data[t] if is_teacher else top1).to(self.device)

        return sentence, hidden_states

    def decode(self, features, captions=None, max_caption_len=30, teacher_forcing_ratio=1):
        batch_size = features.shape[0]

        hidden = self._init_hidden(batch_size)
        outputs, decoder_hiddens = self.forward_sentence(features, captions, hidden, max_caption_len, teacher_forcing_ratio)

        return outputs, decoder_hiddens


    def beam_search_predict(self, features, vocab, max_caption_len=30, beam_alpha=0, beam_width=5):
        batch_size = features.size(0)
        feats = features
        alpha = beam_alpha
        width = beam_width
        rnn_type = self.rnn_type
        vocab_size = len(vocab)
        SOS_idx = vocab.stoi['<SOS>']
        EOS_idx = vocab.stoi['<EOS>']

        hidden = self._init_hidden(batch_size)

        input_list = [ torch.LongTensor(1, batch_size).fill_(SOS_idx).to(self.device) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).to(self.device) ]
        cum_prob_list = [ torch.log(cum_prob).to(self.device) for cum_prob in cum_prob_list ]
        output_list = [ [[]] for _ in range(batch_size) ]

        for t in range(max_caption_len + 1):
            beam_output_list = [] # width x ( 1, 100 )
            normalized_beam_output_list = [] # width x ( 1, 100 )
            if rnn_type == "LSTM":
                beam_hidden_list = ( [], [] ) # 2 * width x ( 1, 100, 512 )
            else:
                beam_hidden_list = [] # width x ( 1, 100, 512 )
            next_output_list = [ [] for _ in range(batch_size) ]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (prev_words, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                
                # print(feats.device, prev_words.device, hidden[0].device)

                output, next_hidden, _ = self.forward_word(feats, hidden, prev_words)

                caption_list = [ output_list[b][i] for b in range(batch_size)]
                EOS_mask = [ 0. if EOS_idx in [ idx.item() for idx in caption ] else 1. for caption in caption_list ]
                EOS_mask = torch.FloatTensor(EOS_mask).to(self.device)
                EOS_mask = EOS_mask.unsqueeze(1).expand_as(output)
                output = EOS_mask * output

                output += cum_prob.unsqueeze(1).to(self.device)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t + 1 for caption in caption_list ]
                caption_lens = torch.FloatTensor(caption_lens).to(self.device)
                normalizing_factor = ((5 + caption_lens) ** alpha) / (6 ** alpha)
                normalizing_factor = normalizing_factor.unsqueeze(1).expand_as(output)
                normalized_output = output / normalizing_factor
                normalized_beam_output_list.append(normalized_output)
                if rnn_type == "LSTM":
                    beam_hidden_list[0].append(next_hidden[0])
                    beam_hidden_list[1].append(next_hidden[1])
                else:
                    beam_hidden_list.append(next_hidden)
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            if rnn_type == "LSTM":
                topk_hidden_list = (
                    [ [] for _ in range(width) ],
                    [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            else:
                topk_hidden_list = [ [] for _ in range(width) ] # width * ( 1, 100, 512 )
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    if rnn_type == "LSTM":
                        topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                        topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    else:
                        topk_hidden_list[k].append(beam_hidden_list[bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            if rnn_type == "LSTM":
                hidden_list = (
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                    [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
                hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            else:
                hidden_list = [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list ] # width * ( 1, 100, 512 )
            cum_prob_list = [ torch.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        return outputs

