import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .temporal_attention import TemporalAttention

class LocalReconstructor(nn.Module):
    def __init__(
        self, 
        decoder_size, ## vocab_size
        hidden_size,  ## feature_dim
        rnn_type="LSTM",
        rnn_num_layers=1,
        rnn_birectional=False,
        rnn_dropout=0.5,
        attn_size = 128, 
        device = 'cpu',
        **args,
    ):
        super(LocalReconstructor, self).__init__()
        self._type = 'local'
        self.rnn_type = rnn_type
        self.num_layers = rnn_num_layers
        self.num_directions = 2 if rnn_birectional else 1
        self.decoder_size = decoder_size
        self.hidden_size = hidden_size
        self.rnn_dropout_p = rnn_dropout if self.num_layers > 1 else 0
        self.attn_size = attn_size
        self.device = device

        RNN = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_size=self.decoder_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False)

        self.attention = TemporalAttention(
            hidden_size=self.num_layers * self.num_directions * self.hidden_size,
            feature_size=self.decoder_size,
            bottleneck_size=self.attn_size)

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

    def get_last_hidden(self, hidden):
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        last_hidden = last_hidden.view(self.num_layers, self.num_directions, last_hidden.size(1), last_hidden.size(2))
        last_hidden = last_hidden.transpose(2, 1).contiguous()
        last_hidden = last_hidden.view(self.num_layers, last_hidden.size(1), self.num_directions * last_hidden.size(3))
        last_hidden = last_hidden[-1]
        return last_hidden

    def reconstruct_single(self, decoder_hiddens, hidden, caption_masks):
        last_hidden = self.get_last_hidden(hidden)
        attention_masks = caption_masks.transpose(0, 1)
        decoder_hidden, attn_weights = self.attention(last_hidden, decoder_hiddens, attention_masks)

        decoder_hidden = decoder_hidden.unsqueeze(0)
        output, hidden = self.rnn(decoder_hidden, hidden)
        return output, hidden

    def reconstruct_sequence(self, decoder_hiddens, caption_masks, feat_len):
        batch_size = decoder_hiddens.shape[2]

        decoder_hiddens = decoder_hiddens.permute(2, 0, 1, 3)
        decoder_hiddens = decoder_hiddens.view(
            decoder_hiddens.size(0),
            decoder_hiddens.size(1),
            decoder_hiddens.size(2) * decoder_hiddens.size(3))

        feats_recons = Variable(torch.zeros(feat_len, batch_size, self.hidden_size))
        feats_recons = feats_recons.to(self.device)
        hidden = self._init_hidden(batch_size)
        
        for t in range(feat_len):
            _, hidden = self.reconstruct_single(decoder_hiddens, hidden, caption_masks)
            feats_recons[t] = hidden[0] if self.rnn_type == 'LSTM' else hidden
        feats_recons = feats_recons.transpose(0, 1)
        return feats_recons

    def reconstruct(self, decoder_hiddens, outputs, captions, target_feature_length):
        caption_masks = build_caption_mask(outputs, captions)
        feats_recons = self.reconstruct_sequence(decoder_hiddens, caption_masks, target_feature_length)
        return feats_recons

class GlobalReconstructor(nn.Module):
    def __init__(
        self, 
        decoder_size, ## vocab_size
        hidden_size,  ## feature_dim
        rnn_type="LSTM",
        rnn_num_layers=1,
        rnn_birectional=False,
        rnn_dropout=0.5,
        device = 'cpu',
        **args,
    ):
        super(GlobalReconstructor, self).__init__()
        self._type = 'global'
        self.rnn_type = rnn_type
        self.num_layers = rnn_num_layers
        self.num_directions = 2 if rnn_birectional else 1
        self.decoder_size = decoder_size
        self.hidden_size = hidden_size
        self.rnn_dropout_p = rnn_dropout if self.num_layers > 1 else 0
        self.device = device

        RNN = nn.LSTM if self.rnn_type == 'LSTM' else nn.GRU
        self.rnn = RNN(
            input_size=self.decoder_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.rnn_dropout_p,
            bidirectional=True if self.num_directions == 2 else False)

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

    def mean_pool_hiddens(self, hiddens, caption_masks):
        caption_lens = caption_masks.sum(dim=0)
        caption_masks = caption_masks.unsqueeze(2).expand_as(hiddens).type_as(hiddens)
        hiddens_masked = caption_masks * hiddens
        hiddens_mean_pooled = hiddens_masked.sum(dim=0) / \
            caption_lens.unsqueeze(1).expand(caption_lens.size(0), hiddens_masked.size(2))
        return hiddens_mean_pooled

    def reconstruct_single(self, decoder_hidden, decoder_hiddens_mean_pooled, hidden):
        input_combined = torch.cat([
            decoder_hidden,
            decoder_hiddens_mean_pooled ], dim=1)
        input_combined = input_combined.unsqueeze(0)

        output, hidden = self.rnn(input_combined, hidden)
        return output, hidden

    def reconstruct_sequence(self, decoder_hiddens, caption_masks):
        
        max_caption_len = decoder_hiddens.shape[0]
        batch_size = decoder_hiddens.shape[2]

        # transform rnn_hidden [len, layer, batch, dim] -> [len, batch, layer*dim]
        decoder_hiddens = decoder_hiddens.transpose(1, 2)
        decoder_hiddens = decoder_hiddens.view(
            decoder_hiddens.size(0),
            decoder_hiddens.size(1),
            decoder_hiddens.size(2) * decoder_hiddens.size(3))

        # mean pooling along length dimension -> [batch, layer*dim]
        decoder_hiddens_mean_pooled = self.mean_pool_hiddens(decoder_hiddens, caption_masks)

        # placeholder for reconstruct features
        feats_recons = Variable(torch.zeros(max_caption_len, batch_size, self.hidden_size))
        feats_recons.to(self.device)

        hidden = self._init_hidden(batch_size)

        for t in range(1, max_caption_len):
            ## reconstruct next feature
            decoder_hidden = decoder_hiddens[t]
            _, hidden = self.reconstruct_single(decoder_hidden, decoder_hiddens_mean_pooled, hidden)
            ## update result
            feats_recons[t] = hidden[0] if self.rnn_type == 'LSTM' else hidden

        feats_recons = feats_recons.transpose(0, 1)
        return feats_recons

    def reconstruct(self, decoder_hiddens, outputs, captions, target_feature_length=None):
        '''
        target_feature_length is useless, just to standardize the function call with global_reconstructor
        '''
        caption_masks = build_caption_mask(outputs, captions)
        feats_recons = self.reconstruct_sequence(decoder_hiddens, caption_masks)
        return feats_recons
        

def build_caption_mask(outputs, captions=None):
    '''
    outputs and captions is used for build mask
    use caption if captions is available, else take argmax from outputs
    '''
    if captions is None:
        _, captions = outputs.max(dim=2)
    caption_masks = (captions != 0) * (captions != 2)  # self.vocab.stoi['<PAD>'] = 0, self.vocab.stoi['<EOS>'] = 2
    caption_masks = caption_masks.to(outputs.device)
    return caption_masks
