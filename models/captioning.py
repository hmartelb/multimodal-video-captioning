import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .temporal_attention import TemporalAttention
from .features_captioning import FeaturesCaptioning
from .reconstructor import GlobalReconstructor, LocalReconstructor



DECODER_CONFIG = {
    'rnn_type'       : 'LSTM', # ['LSTM', 'GRU']
    'rnn_num_layers' : 1,
    'rnn_bidirectional': False,  # Bool
    'rnn_hidden_size': 512,
    'rnn_dropout'    : 0.5,    
    
    'in_feature_size': 1000+128,
    'embedding_size' : 128,
    'attn_size'      : 128,
    'output_size'    : 3201, #Vocab Size
}

RECONSTRUCTOR_CONFIG = {   
    'type'           : 'global',  # ['none', 'global', 'local']
    'rnn_type'       : 'LSTM',    # ['LSTM', 'GRU']
    'rnn_num_layers' : 1,
    'rnn_bidirectional': False,     # Bool
    'hidden_size'    : 1000+128,  # feature_size
    'rnn_dropout'    : 0.5,    
    'decoder_size'   : 128,       # decoder_hidden_size
    'attn_size'      : 128,       # only applied for local
}


class AVCaptioning(nn.Module):
    def __init__(
        self,
        vocab,
        teacher_forcing_ratio=0.0,
        no_reconstructor = False,
        device='cpu',
    ):
        super(AVCaptioning, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        config = DECODER_CONFIG.copy()
        config['output_size'] = self.vocab_size

        rec_config = RECONSTRUCTOR_CONFIG.copy()
        rec_config['decoder_size'] = config['rnn_hidden_size']
        rec_config['hidden_size'] = config['in_feature_size']

        decoder = FeaturesCaptioning(**config, device=device)
        self.decoder = decoder.to(device)

        if no_reconstructor:
            rec_config['type'] = 'none'

        if rec_config['type'] == "global":
            reconstructor = GlobalReconstructor(**rec_config, device=device)
            self.reconstructor = reconstructor.to(device)
        elif rec_config['type'] == "local":
            reconstructor = LocalReconstructor(**rec_config, device=device)
            self.reconstructor = reconstructor.to(device)
        else:
            self.reconstructor = None

        self.reconstructor_type = rec_config['type']

        ## Message
        print("Initializing Model...")
        print(
            "Decoder      :", config['rnn_type'], 
            'In:', config['in_feature_size'], 
            'Out:', config['output_size'],
            'Hidden:', config['rnn_hidden_size']
        )
        print("Reconstuctor :", rec_config['type'])
        
    def forward(self, audio_features, visual_features, captions):
        features = torch.cat([audio_features, visual_features], dim=-1)
        outputs, rnn_hiddens = self.decoder.decode(
            features, captions, 
            max_caption_len=captions.shape[0], 
            teacher_forcing_ratio=self.teacher_forcing_ratio
        )

        if self.reconstructor is None:
            features_recons = None
            audio_recons = None
            visual_recons = None
        else:
            features_recons = self.reconstructor.reconstruct(rnn_hiddens, outputs, captions, features.shape[1])
            audio_recons = features_recons[:, :, 0:audio_features.shape[2]]
            visual_recons = features_recons[:, :, audio_features.shape[2]:]

        return outputs, audio_recons, visual_recons
        # return outputs, features_recons

    def predict(self, features, max_caption_len=30, beam_alpha=0, beam_width=5):
        outputs = self.decoder.beam_search_predict(features, self.vocab, max_caption_len, beam_alpha, beam_width)

        captions = [self.vocab.decode_indexes(o[1:]) for o in outputs]

        return captions
