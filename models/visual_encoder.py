import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class VisualEncoder(nn.Module):
    def __init__(
        self,
        # embedding_dim,
        model="inception_v3",
        normalize=True,
        trainable=False,
    ):
        super(VisualEncoder, self).__init__()
        self.normalize = (
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else nn.Identity()
        )
        self.trainable = trainable

        if model == "resnet":
            self.extractor = models.resnet101(pretrained=True)
            self.input_size = (224, 224)

        if model == "inception_v3":
            self.extractor = models.inception_v3(pretrained=True, aux_logits=False)
            self.extractor.fc = nn.Identity()#nn.Linear(self.extractor.fc.in_features, 128) # #
            self.input_size = (299, 299)

        # if embedding_dim != self.extractor.fc.out_features:
        #     self.linear = nn.Linear(self.extractor.fc.in_features, embedding_dim)
        # else:
        # self.linear = nn.Identity()

        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)

        if not self.trainable:
            self.extractor.eval()

    def forward(self, frames):
        stack_frames = frames.dim() == 5
        frames_shape = frames.shape

        # Make batch videos into batch of images (all frames of all videos stacked)
        if stack_frames:
            frames = frames.view(frames_shape[0] * frames_shape[1], frames_shape[2], frames_shape[3], frames_shape[4])

        frames = self.normalize(frames)
        features = self.extractor(frames)

        # Undo stacking operation
        if stack_frames:
            features = features.view(frames_shape[0], frames_shape[1], features.shape[1])

        return features
        # embedding = self.linear(features)
        # return embedding
        # return self.dropout(self.relu(embedding))