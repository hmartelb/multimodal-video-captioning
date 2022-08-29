import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
        self, model="vggish", trainable=False,
    ):
        super(AudioEncoder, self).__init__()
        self.trainable = trainable

        if model == "vggish":
            self.extractor = torch.hub.load("harritaylor/torchvggish", "vggish")

        if not self.trainable:
            self.extractor.eval()

    def forward(self, audio, sr=16000):
        features = self.extractor(audio, fs=sr)
        return features

    def from_file(self, filename):
        features = self.forward(filename)
        return features

    def from_tensor(self, audio, sr, fmt="channels_first"):
        if fmt == "channels_first":
            audio = audio.permute(1, 0).numpy()
        features = self.forward(audio, sr)
        return features
