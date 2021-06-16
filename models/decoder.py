import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .temporal_attention import TemporalAttention


class Decoder(nn.Module):
    def __init__(
        self,
        rnn_type="LSTM",
        num_layers=1,
        num_directions=1,
        feature_size=1128,
        embedding_size=128,
        hidden_size=128,
        attn_size=128,
        output_size=3056,
        rnn_dropout=0.5,
        max_caption_len=30,
        teacher_forcing_ratio=0.0,
    ):
        super(Decoder, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.feature_size = feature_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.output_size = output_size
        self.rnn_dropout_p = rnn_dropout

        self.max_caption_len = max_caption_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)

        # Apply Soft-Attention (SA) ?
        # if self.attn_size > 0:
        self.attention = TemporalAttention(
            hidden_size=self.num_directions * self.hidden_size,
            feature_size=self.feature_size,
            bottleneck_size=self.attn_size,
        )
        # else:
        #     self.attention = nn.Identity() # FIXME: 
        #          
        #       features, attn_weights = self.attention(last_hidden, features)
        #       -- result = self.forward(*input, **kwargs)
        #       TypeError: forward() takes 2 positional arguments but 3 were given
        #

        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(
            input_size=self.embedding_size + self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False,
        )

        self.out = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def _init_hidden(self, batch_size):
        if self.rnn_type == "LSTM":
            hidden = (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).cuda(),
            )
        else:
            hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden = hidden.cuda()
        return hidden

    def _last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), self.num_directions * last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden

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

    def forward_sentence(self, features, captions, hidden):
        batch_size = features.shape[0]  # Shape [batch_size, n_seconds, feature_size]

        # Placeholoder variables initialization
        # (max_caption_len + 2 because we add 2 special tokens: <SOS> and <EOS>)
        sentence = Variable(torch.zeros(self.max_caption_len, batch_size, self.output_size)).cuda()
        D, B, H = (hidden[0] if self.rnn_type == "LSTM" else hidden).shape
        hidden_states = Variable(torch.zeros(self.max_caption_len, D, B, H)).cuda()
        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(1))  # self.vocab.stoi["<SOS>"]

        for t in range(1, self.max_caption_len):
            # Get the next word
            output, hidden, attn_weights = self.forward_word(features, hidden, output.view(1, -1))

            # Update result and hidden state for next prediction
            sentence[t] = output
            hidden_states[t] = hidden[0] if self.rnn_type == "LSTM" else hidden

            # Apply teacher forcing ?
            # Randomly decide whether or not to apply it
            # YES: take true word, NO: take prediction
            is_teacher = torch.rand(1) < self.teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(captions.data[t] if is_teacher else top1).cuda()

        return sentence, hidden_states

    def decode(self, features, captions=None, max_caption_len=None):
        if max_caption_len: 
            self.max_caption_len = max_caption_len

        batch_size = features.shape[0]

        hidden = self._init_hidden(batch_size)
        outputs, decoder_hiddens = self.forward_sentence(features, captions, hidden)

        if captions is None:
            _, captions = outputs.max(dim=2)
        caption_masks = (captions != 0) * (captions != 2)  # self.vocab.stoi['<PAD>'] = 0, self.vocab.stoi['<EOS>'] = 2
        caption_masks = caption_masks.cuda()

        recons = None
        # if self.reconstructor is not None:
        #     recons = self.forward_reconstructor(batch_size, decoder_hiddens, caption_masks)
        return outputs, recons


if __name__ == "__main__":

    batch_size = 32
    features, captions = torch.rand([batch_size, 10, 1128]).cuda(), torch.rand([8, batch_size]).cuda()

    model = Decoder(
        output_size=3056,
        attn_size=128,
        max_caption_len=10,
    )
    model = model.cuda()

    output, recons = model.decode(features, captions)
        
    print(output.shape)
        