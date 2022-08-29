import torch
import torch.nn as nn

from .features_captioning import FeaturesCaptioning
from .reconstructor import GlobalReconstructor, LocalReconstructor

DECODER_CONFIG = {
    "rnn_type": "LSTM",  # ['LSTM', 'GRU']
    "rnn_num_layers": 1,
    "rnn_bidirectional": False,  # Bool
    "rnn_hidden_size": 512,
    "rnn_dropout": 0.0,  # 0.5,
    "in_feature_size": 2048 + 128,
    "embedding_size": 300,
    "attn_size": 256,
    "output_size": 1024,  # Vocab Size
}

RECONSTRUCTOR_CONFIG = {
    "type": "global",  # ['none', 'global', 'local']
    "rnn_type": "LSTM",  # ['LSTM', 'GRU']
    "rnn_num_layers": 1,
    "rnn_bidirectional": False,  # Bool
    "hidden_size": 2048 + 128,  # feature_size
    "rnn_dropout": 0.5,
    "decoder_size": 512,  # decoder_hidden_size
    "attn_size": 256,  # only applied for local
}

#
# For the DUAL model
#
VISUAL_DECODER_CONFIG = {
    "rnn_type": "LSTM",  # ['LSTM', 'GRU']
    "rnn_num_layers": 1,
    "rnn_bidirectional": False,  # Bool
    "rnn_hidden_size": 512,
    "rnn_dropout": 0.0,  # 0.5,
    "in_feature_size": 2048,
    "embedding_size": 300,
    "attn_size": 256,
    "output_size": 1024,  # Vocab Size
}

AUDIO_DECODER_CONFIG = {
    "rnn_type": "LSTM",  # ['LSTM', 'GRU']
    "rnn_num_layers": 1,
    "rnn_bidirectional": False,  # Bool
    "rnn_hidden_size": 512,
    "rnn_dropout": 0.0,  # 0.5,
    "in_feature_size": 128,
    "embedding_size": 300,
    "attn_size": 256,
    "output_size": 512,  # Vocab Size
}


class AVCaptioning(nn.Module):
    def __init__(
        self, vocab, teacher_forcing_ratio=0.0, reconstructor_type="none", device="cpu", normalize_inputs=False,
    ):
        super(AVCaptioning, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.normalize_inputs = normalize_inputs

        config = DECODER_CONFIG.copy()
        config["output_size"] = self.vocab_size

        rec_config = RECONSTRUCTOR_CONFIG.copy()
        rec_config["decoder_size"] = config["rnn_hidden_size"]
        rec_config["hidden_size"] = config["in_feature_size"]
        rec_config["type"] = reconstructor_type

        decoder = FeaturesCaptioning(**config, device=device)
        self.decoder = decoder.to(device)

        # if no_reconstructor:
        #     rec_config["type"] = "none"

        if reconstructor_type == "global":
            reconstructor = GlobalReconstructor(**rec_config, device=device)
            self.reconstructor = reconstructor.to(device)
        elif reconstructor_type == "local":
            reconstructor = LocalReconstructor(**rec_config, device=device)
            self.reconstructor = reconstructor.to(device)
        else:
            self.reconstructor = None

        self.reconstructor_type = reconstructor_type  # rec_config["type"]

        ## Message
        print("Initializing Model...")
        print(
            "Decoder      :",
            config["rnn_type"],
            "In:",
            config["in_feature_size"],
            "Out:",
            config["output_size"],
            "Hidden:",
            config["rnn_hidden_size"],
        )
        print("Reconstuctor :", rec_config["type"])

    def forward(self, audio_features, visual_features, captions, teacher_forcing_ratio=None):
        features = torch.cat([audio_features, visual_features], dim=-1)
        outputs, rnn_hiddens = self.decoder.decode(
            features,
            captions,
            max_caption_len=captions.shape[0],
            teacher_forcing_ratio=teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.teacher_forcing_ratio,
        )

        if self.reconstructor is None:
            features_recons = None
            audio_recons = None
            visual_recons = None
        else:
            features_recons = self.reconstructor.reconstruct(rnn_hiddens, outputs, captions, features.shape[1])
            audio_recons = features_recons[:, :, 0 : audio_features.shape[2]]
            visual_recons = features_recons[:, :, audio_features.shape[2] :]

        return outputs, audio_recons, visual_recons
        # return outputs, features_recons

    def predict(self, audio_features, visual_features, max_caption_len=30, mode="direct", beam_alpha=0, beam_width=5):
        features = torch.cat([audio_features, visual_features], dim=-1)

        if mode == "beam":
            outputs = self.decoder.beam_search_predict(features, self.vocab, max_caption_len, beam_alpha, beam_width)

        if mode == "direct":
            outputs, _ = self.decoder.decode(features, captions=None, max_caption_len=max_caption_len)
            # outputs > [max_caption_len, batch_size, vocab_size]
            outputs = outputs.argmax(2).transpose(1, 0)
            # outputs > [batch_size, max_caption_len]

        captions = [self.vocab.decode_indexes(o[1:]) for o in outputs]
        return captions


class AVCaptioningDual(nn.Module):
    def __init__(
        self, vocab, teacher_forcing_ratio=0.0, reconstructor_type="none", device="cpu", normalize_inputs=False,
    ):
        super(AVCaptioningDual, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.normalize_inputs = normalize_inputs

        v_config = VISUAL_DECODER_CONFIG.copy()
        v_config["output_size"] = self.vocab_size

        a_config = AUDIO_DECODER_CONFIG.copy()
        a_config["output_size"] = self.vocab_size

        ## FIXME: temporall setup
        VISUAL_RECONSTRUCTOR_CONFIG = RECONSTRUCTOR_CONFIG
        AUDIO_RECONSTRUCTOR_CONFIG = RECONSTRUCTOR_CONFIG
        ##
        v_rec_config = VISUAL_RECONSTRUCTOR_CONFIG.copy()
        v_rec_config["decoder_size"] = v_config["rnn_hidden_size"]
        v_rec_config["hidden_size"] = v_config["in_feature_size"]
        v_rec_config["type"] = reconstructor_type

        a_rec_config = AUDIO_RECONSTRUCTOR_CONFIG.copy()
        a_rec_config["decoder_size"] = a_config["rnn_hidden_size"]
        a_rec_config["hidden_size"] = a_config["in_feature_size"]
        a_rec_config["type"] = reconstructor_type

        v_decoder = FeaturesCaptioning(**v_config, device=device)
        self.v_decoder = v_decoder.to(device)

        a_decoder = FeaturesCaptioning(**a_config, device=device)
        self.a_decoder = a_decoder.to(device)

        # output fusion
        self.output_fc = nn.Linear(a_config["output_size"] + v_config["output_size"], self.vocab_size)

        # if no_reconstructor:
        #     rec_config["type"] = "none"

        if reconstructor_type == "global":
            v_reconstructor = GlobalReconstructor(**v_rec_config, device=device)
            self.v_reconstructor = v_reconstructor.to(device)

            a_reconstructor = GlobalReconstructor(**a_rec_config, device=device)
            self.a_reconstructor = a_reconstructor.to(device)

        elif reconstructor_type == "local":
            v_reconstructor = LocalReconstructor(**v_rec_config, device=device)
            self.v_reconstructor = v_reconstructor.to(device)

            a_reconstructor = LocalReconstructor(**a_rec_config, device=device)
            self.a_reconstructor = a_reconstructor.to(device)
        else:
            self.v_reconstructor = None
            self.a_reconstructor = None

        self.reconstructor_type = reconstructor_type  # rec_config["type"]

        ## Message
        print("Initializing Model...")
        print(
            "Decoder (V,A)     :",
            (v_config["rnn_type"], a_config["rnn_type"]),
            "In:",
            (v_config["in_feature_size"], a_config["in_feature_size"]),
            "Out:",
            (v_config["output_size"], a_config["output_size"]),
            "Hidden:",
            (v_config["rnn_hidden_size"], a_config["rnn_hidden_size"]),
        )
        print("Reconstuctor (V,A):", (v_rec_config["type"], a_rec_config["type"]))

    def forward(self, audio_features, visual_features, captions, teacher_forcing_ratio=None):
        # features = torch.cat([audio_features, visual_features], dim=-1)
        v_outputs, v_rnn_hiddens = self.v_decoder.decode(
            visual_features,
            captions,
            max_caption_len=captions.shape[0],
            teacher_forcing_ratio=teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.teacher_forcing_ratio,
        )
        a_outputs, a_rnn_hiddens = self.a_decoder.decode(
            audio_features,
            captions,
            max_caption_len=captions.shape[0],
            teacher_forcing_ratio=teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.teacher_forcing_ratio,
        )

        # Perform the concatenation and FC of each modality output
        outputs = self._feature_fusion(a_outputs, v_outputs)

        if self.a_reconstructor is None:
            audio_recons = None
        else:
            audio_recons = self.a_reconstructor.reconstruct(a_rnn_hiddens, a_outputs, captions, audio_features.shape[1])

        if self.v_reconstructor is None:
            visual_recons = None
        else:
            visual_recons = self.v_reconstructor.reconstruct(
                v_rnn_hiddens, v_outputs, captions, visual_features.shape[1]
            )

        return outputs, audio_recons, visual_recons
        # return outputs, features_recons

    def _feature_fusion(self, a_outputs, v_outputs):
        """
        Simply fuse the audio and video features with summation. Overwrite this method for other operations.
        """
        return a_outputs + v_outputs

    def predict(
        self, audio_features, visual_features, max_caption_len=30, mode="direct", beam_alpha=0, beam_width=5,
    ):
        if mode == "beam":
            v_outputs = self.v_decoder.beam_search_predict(
                visual_features, self.vocab, max_caption_len, beam_alpha, beam_width
            )
            a_outputs, _ = self.a_decoder.beam_search_predict(
                audio_features, self.vocab, max_caption_len, beam_alpha, beam_width
            )

            # FIXME: not implemented

        if mode == "direct":
            v_outputs, _ = self.v_decoder.decode(visual_features, captions=None, max_caption_len=max_caption_len)
            a_outputs, _ = self.a_decoder.decode(audio_features, captions=None, max_caption_len=max_caption_len)

            # Perform the concatenation and FC of each modality output
            outputs = self._feature_fusion(a_outputs, v_outputs)

            # outputs > [max_caption_len, batch_size, vocab_size]
            outputs = outputs.argmax(2).transpose(1, 0)
            # outputs > [batch_size, max_caption_len]

        captions = [self.vocab.decode_indexes(o[1:]) for o in outputs]
        return captions

