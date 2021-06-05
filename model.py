import statistics

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
# import torchvision.transforms as trasnforms

class AudioEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        model="vggish",
        trainable=False,
    ):
        super(AudioEncoder, self).__init__()
        self.trainable = trainable

        if model == "vggish":
            self.extractor = torch.hub.load("harritaylor/torchvggish", "vggish")
            self.linear = nn.Linear(128, embedding_dim)

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
        embedding_dim,
        model="inception_v3",
        transforms=None,
        trainable=False,
    ):
        super(ImageEncoder, self).__init__()
        self.transforms = transforms
        self.trainable = trainable

        if model == "resnet":
            self.extractor = models.resnet101(pretrained=True)
            self.extractor.fc = nn.Linear(self.extractor.fc.in_features, embedding_dim)
            # self.transforms = torchvision.transforms.Compose(
            #     [
            #         torchvision.transforms.Resize((224, 224)),
            #         torchvision.transforms.ToTensor(),
            #     ]
            # )

        if model == "inception_v3":
            self.extractor = models.inception_v3(pretrained=True, aux_logits=False)
            self.extractor.fc = nn.Linear(self.extractor.fc.in_features, embedding_dim)
            # self.transforms = torchvision.transforms.Compose(
            #     [
            #         torchvision.transforms.Resize((299, 299)),
            #         torchvision.transforms.ToTensor(),
            #     ]
            # )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # if self.transforms:
        #     images = self.transforms(images)
        
        features = self.extractor(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        vocab_size,
        num_layers,
    ):
        super(DecoderRNN, self).__init__()
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
