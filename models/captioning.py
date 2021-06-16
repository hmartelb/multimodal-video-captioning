import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .temporal_attention import TemporalAttention
from .features_captioning import FeaturesCaptioning
from .reconstructor import GlobalReconstructor, LocalReconstructor



decoder_config = {    
    'rnn_type'       : 'LSTM', # ['LSTM', 'GRU']
    'rnn_num_layers' : 1,
    'rnn_birectional': False,  # Bool
    'rnn_hidden_size': 512,
    'rnn_dropout'    : 0.5,    
    
    'in_feature_size': 1000+128,
    'embedding_size' : 128,
    'attn_size'      : 128,
    'output_size'    : 3201, #Vocab Size

    'rnn_teacher_forcing_ratio' : 1.0,
    'max_caption_len' : 30,
}

constructor_config = {   
    'type'           : 'global',  # ['none', 'global', 'local']
    'rnn_type'       : 'LSTM',    # ['LSTM', 'GRU']
    'rnn_num_layers' : 1,
    'rnn_birectional': False,     # Bool
    'hidden_size'    : 512,       # feature_size
    'rnn_dropout'    : 0.5,    
    'decoder_size'   : 128,       # decoder_hidden_size
    'attn_size'      : 128,       # only applied for local
}


class AVCaptioning(nn.Module):
    def __init__(
        self,
        vocab_size,
        teacher_forcing_ratio=0.0,
        device='cpu',
    ):
        super(AVCaptioning, self).__init__()
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        config = decoder_config.copy()
        config['output_size'] = vocab_size

        decoder = FeaturesCaptioning(**config,device=device)
        self.decoder = decoder.to(device)

        rec_config = constructor_config.copy()
        rec_config['decoder_size'] = config['rnn_hidden_size']
        rec_config['hidden_size'] = config['in_feature_size']
        if constructor_config['type'] == "global":
            reconstructor = GlobalReconstructor(**rec_config,device=device)
            self.reconstructor = reconstructor.to(device)
        elif constructor_config['type'] == "local":
            reconstructor = LocalReconstructor(**rec_config,device=device)
            self.reconstructor = reconstructor.to(device)
        else:
            self.reconstructor = None

        self.reconstructor_type = constructor_config['type']
        
    def forward(self, features, captions):
        outputs, rnn_hiddens = self.decoder.decode(features, captions, max_caption_len=captions.shape[0])

        if self.reconstructor is None:
            features_recons = None
        else:
            features_recons = self.reconstructor.reconstruct(rnn_hiddens, outputs, captions, features.shape[1])

        return outputs, features_recons
    