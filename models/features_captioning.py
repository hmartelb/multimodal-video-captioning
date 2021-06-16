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
        rnn_birectional=False,
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
        self.num_directions = 2 if rnn_birectional else 1
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

