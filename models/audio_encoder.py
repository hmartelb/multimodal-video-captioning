import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class AudioEncoder(nn.Module):
    def __init__(
        self,
        # embedding_dim,
        model="vggish",
        trainable=False,
    ):
        super(AudioEncoder, self).__init__()
        self.trainable = trainable

        if model == "vggish":
            self.extractor = torch.hub.load("harritaylor/torchvggish", "vggish")
            # if embedding_dim != 128:
            #     self.linear = nn.Linear(128, embedding_dim)
            # else:
        #     self.linear = nn.Identity()

        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)

        if not self.trainable:
            self.extractor.eval()

    def forward(self, audio, sr=16000):
        features = self.extractor(audio, fs=sr)
        return features
        # features = self.linear(features)
        # return self.dropout(self.relu(features))

    def from_file(self, filename):
        features = self.forward(filename)
        return features

    def from_tensor(self, audio, sr, fmt="channels_first"):
        if fmt == "channels_first":
            audio = audio.permute(1, 0).numpy()
        features = self.forward(audio, sr)
        return features
