import statistics

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

# import torchvision.transforms as trasnforms


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
            self.linear = nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, audio, sr=16000):
        features = self.extractor(audio, fs=sr)
        features = self.linear(features)
        return self.dropout(self.relu(features))

    def from_file(self, filename):
        features = self.forward(filename)
        return features

    def from_tensor(self, audio, sr, fmt="channels_first"):
        if fmt == "channels_first":
            audio = audio.permute(1, 0).numpy()
        features = self.forward(audio, sr)
        return features


class ImageEncoder(nn.Module):
    def __init__(
        self,
        # embedding_dim,
        model="inception_v3",
        transforms=None,
        trainable=False,
    ):
        super(ImageEncoder, self).__init__()
        self.transforms = transforms
        self.trainable = trainable

        if model == "resnet":
            self.extractor = models.resnet101(pretrained=True)
            self.input_size = (224, 224)

        if model == "inception_v3":
            self.extractor = models.inception_v3(pretrained=True, aux_logits=False)
            self.input_size = (299, 299)

        # if embedding_dim != self.extractor.fc.out_features:
        #     self.linear = nn.Linear(self.extractor.fc.in_features, embedding_dim)
        # else:
        self.linear = nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if not self.trainable:
            self.extractor.eval()

    def forward(self, frames):
        stack_frames = frames.dim() == 5
        frames_shape = frames.shape

        # Make batch videos into batch of images (all frames of all videos stacked)
        if stack_frames:
            frames = frames.view(frames_shape[0] * frames_shape[1], frames_shape[2], frames_shape[3], frames_shape[4])

        features = self.extractor(frames)

        # Undo stacking operation
        if stack_frames:
            features = features.view(frames_shape[0], frames_shape[1], features.shape[1])

        embedding = self.linear(features)
        return self.dropout(self.relu(embedding))


class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab,
        embedding_dim=128,
        hidden_size=128,
        num_layers=1,
    ):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(self.vocab)
      
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def generate_caption(self, features, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = features  # Check this
            states = None

            for _ in range(max_length):
                hiddens, states = self.lstm(x, states)
                output = self.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.embed(predicted).unsqueeze(0)

                if self.vocab.itos[predicted.item()] == "<EOS>":
                    break

        return [self.vocab.itos[idx] for idx in result_caption]


class VisualCaptioningModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        vocab_size,
        num_layers,
    ):
        super(VisualCaptioningModel, self).__init__()
        self.v_encoder = ImageEncoder(embedding_dim)
        self.decoder = DecoderRNN(embedding_dim, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.v_encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, video_frames, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.v_encoder(video_frames).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


if __name__ == "__main__":

    pass
