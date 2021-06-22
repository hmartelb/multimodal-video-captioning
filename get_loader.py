import os  # when loading file paths

import numpy as np
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
import torchvision.transforms as transforms
from PIL import Image  # Load img
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
import pickle
import json

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    @staticmethod
    def prebuild(sentence_list, outpath, freq_threshold=5):
        vocab = Vocabulary(freq_threshold)
        vocab.build_vocabulary(sentence_list)
        vocab.save_vocab(outpath)

    def save_vocab(self, path):
        pickle.dump(self, open(path, "wb+"))
        print(f"Vocab saved: {path}")

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

    def apply_vocab(self, sentence):
        """
        convert token not in vocab to "<UNK>"
        """
        tokenized_text = self.tokenizer_eng(sentence)
        tokens = [token if token in self.stoi else "<UNK>" for token in tokenized_text]
        return " ".join(tokens)

    def decode_indexes(self, indexes):
        words = []
        EOS_idx = 2
        for idx in indexes:
            if type(idx) is torch.Tensor:
                idx = idx.item()
            if idx == EOS_idx:
                break
            words.append(self.itos[idx])
        sentence = " ".join(words)
        return sentence

def build_MSR_VTT_vocab():
    dataset_folder = os.path.join("datasets", "MSR-VTT")
    json_path = os.path.join(dataset_folder, "metadata", "train_val_videodatainfo.json")
    vocab_path = os.path.join(dataset_folder, "metadata", "vocab.pkl")

    data = json.load(open(json_path, 'r'))
    metadata = pd.DataFrame(data['sentences'])
    metadata['id'] = metadata.video_id.apply(lambda x: int(x.replace('video','')))
    # train: 0-6512, Val: 6513-7009
    sentences = metadata[(metadata.id >= 0) & (metadata.id < 7009)]['caption'].tolist()

    Vocabulary.prebuild(sentences, vocab_path)

def build_MSVD_vocab():
    dataset_folder = os.path.join("datasets", "MSVD")
    train_captions_file = os.path.join(dataset_folder, "metadata", "train.csv")
    val_captions_file = os.path.join(dataset_folder, "metadata", "val.csv")
    train_captions = pd.read_csv(train_captions_file)["Description"].tolist()
    val_captions = pd.read_csv(val_captions_file)["Description"].tolist()
    vocab_path = os.path.join(dataset_folder, "metadata", "vocab.pkl")

    Vocabulary.prebuild(train_captions + val_captions, vocab_path)


def read_MSVD_Metadata(root_dir, split):
    '''
    return pandas data frame in format [video_id, caption]
    '''

    captions_file = os.path.join(root_dir, "metadata", f"{split}.csv")
    assert os.path.isfile(captions_file), f"The captions file cannot be found {captions_file}"

    def read_video_filename(video_name):
        filename = video_name.split(".")[0]
        parts = filename.split("_")
        video_id = '_'.join(parts[:-2])
        start = parts[-2]
        end = parts[-1]
        start, end = int(start), int(end)
        return video_id, start, end

    multimodal_video_ids = set([])
    for f in os.listdir(os.path.join(root_dir, "features", "video")):
        _, ext = os.path.splitext(f)
        video_id, start, end = read_video_filename(f)

        video_filename = os.path.join(root_dir, "features", "video", f)
        # audio_filename = os.path.join(root_dir, "features", "audio", f.replace(ext, '.wav'))

        if os.path.isfile(video_filename):# and os.path.isfile(audio_filename):
            multimodal_video_ids.add(f"{video_id}_{start}_{end}")

    metadata = pd.read_csv(captions_file)
    print("Before integrity check:", len(metadata))

    to_drop = []
    for i, row in metadata.iterrows():
        if f"{row['VideoID']}_{row['Start']}_{row['End']}" not in multimodal_video_ids:
            to_drop.append(i)
    
    metadata = metadata.drop(index=to_drop)
        
    print("After integrity check:", len(metadata))

    metadata = metadata[(metadata['Source'] == 'clean')]

    print("After removing unverified:", len(metadata))
  
    metadata["video_id"] = metadata.apply(lambda x: f"{x['VideoID']}_{x['Start']}_{x['End']}", axis=1)
    metadata = metadata.rename(columns={"Description": "caption"})

    return metadata[["video_id", "caption"]]

def read_MSR_VTT_Metadata(root_dir, split):
    if split is "test":
        json_path = os.path.join(root_dir, "metadata", "test_videodatainfo.json")
    else:
        json_path = os.path.join(root_dir, "metadata", "train_val_videodatainfo.json")

    assert os.path.isfile(json_path), f"The captions file cannot be found {json_path}"

    data = json.load(open(json_path, 'r'))
    metadata = pd.DataFrame(data['sentences'])
    metadata['id'] = metadata.video_id.apply(lambda x: int(x.replace('video','')))
    start_end = {
        "train": [0, 6512], # (6513)
        "val": [6513, 7009], # (497)
        "test": [7010, 9999] # (2990)
    }
    start, end = start_end[split]
    metadata = metadata[(metadata.id >= start) & (metadata.id < end)][['video_id', 'caption']]

    print(f"Total Data Count (MSR-VTT-{split}):", len(metadata))
    return metadata


class VideoCaptioningDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataset="MSVD",
        split="train",
        transform=None,
        freq_threshold=5,
        vocab_pkl=None,
        normalize = False,
        video_only=False,
    ):
        self.video_only = video_only
        self.normalize = normalize
        self.root_dir = root_dir

        assert os.path.isdir(root_dir), "The dataset root directory does not exist"
        assert os.path.isdir(os.path.join(root_dir, "metadata")), "The dataset metadata directory does not exist"
        assert os.path.isdir(os.path.join(root_dir, "features")), "The dataset features directory does not exist"
        assert dataset in ["MSVD", "MSR-VTT"], "Dataset must be one of ['MSVS', 'MSR-VTT']"
        assert split in [
            "train",
            "val",
            "test",
            "tiny",
        ], "Wrong split specified, must be one of ['train', 'val', 'test']"
        if vocab_pkl is not None:
            assert os.path.isfile(vocab_pkl), f"The vocab file cannot be found {vocab_pkl}"

        if dataset == "MSVD":
            self.metadata = read_MSVD_Metadata(root_dir, split)
        elif dataset == "MSR-VTT":
            self.metadata = read_MSR_VTT_Metadata(root_dir, split)


        # Initialize vocabulary and build vocab
        if vocab_pkl is None:
            print("Building Vocab")
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.metadata["caption"].tolist())
        else:
            print(f"Loading Vocab: {vocab_pkl} ")
            self.vocab = Vocabulary.load(vocab_pkl)

        self.features = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        video_id, caption = row["video_id"], row["caption"]
    
        # if video_id not in self.features:
        caption_tokens = [self.vocab.stoi["<SOS>"]]
        caption_tokens += self.vocab.numericalize(caption)
        caption_tokens.append(self.vocab.stoi["<EOS>"])            

        video_features_file = os.path.join(self.root_dir, "features", "video", f"{video_id}.npy")
        audio_features_file = os.path.join(self.root_dir, "features", "audio", f"{video_id}.npy")

        video_features = np.load(video_features_file)
        audio_features = np.load(audio_features_file)

        # quick fix,there are some feature in shape (128,) when number of frame is 1
        # e.g. 'rOic25PnIx8_1_3'
        if len(audio_features.shape) < 2:
            audio_features = audio_features.reshape((-1, 128))

        # Make both features to have the same frames (drop largest)
        n_frames = min(video_features.shape[0], audio_features.shape[0])

        video_features = video_features[0:n_frames, :]
        audio_features = audio_features[0:n_frames, :]

        # Frame-wise normalization
        if self.normalize:
            video_features /= np.sum(video_features, axis=1, keepdims=True)
            audio_features /= np.sum(audio_features, axis=1, keepdims=True)

        # Make zeros in audio features
        if self.video_only:
            audio_features *= 0 

        # [n_seconds, n_feats] -> sum(n_feats) and divide 

        # self.features[video_id] = {
        #     'audio': torch.tensor(audio_features),
        #     'video': torch.tensor(video_features), 
        #     'captions': torch.tensor(caption_tokens)
        # }

        return torch.tensor(audio_features), torch.tensor(video_features), torch.tensor(caption_tokens)
        # return self.features[video_id]['audio'], self.features[video_id]['video'], self.features[video_id]['captions']

        # features = np.concatenate([video_features, audio_features], axis=1)
        # return torch.tensor(features), torch.tensor(caption_tokens)


class VideoCaptionsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        vid_cap_dict,
        normalize = False,
        video_only=False,
    ):
        """
        vid_cap_dict: dict({ vid: [captions] })
        """
        self.video_only = video_only
        self.normalize = normalize
        self.root_dir = root_dir
        self.vid_cap_dict = vid_cap_dict
        self.video_ids = list(vid_cap_dict.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id_full = self.video_ids[index]

        video_features_file = os.path.join(self.root_dir, "features", "video", f"{video_id_full}.npy")
        audio_features_file = os.path.join(self.root_dir, "features", "audio", f"{video_id_full}.npy")

        video_features = np.load(video_features_file)
        audio_features = np.load(audio_features_file)

        # quick fix,there are some feature in shape (128,) when number of frame is 1
        # e.g. 'rOic25PnIx8_1_3'
        if len(audio_features.shape) < 2:
            audio_features = audio_features.reshape((-1, 128))

        # Make both features to have the same frames (drop largest)
        n_frames = min(video_features.shape[0], audio_features.shape[0])

        video_features = video_features[0:n_frames, :]
        audio_features = audio_features[0:n_frames, :]

        # Frame-wise normalization
        if self.normalize:
            video_features /= np.sum(video_features, axis=1, keepdims=True)
            audio_features /= np.sum(audio_features, axis=1, keepdims=True)

        # Make zeros in audio features
        if self.video_only:
            audio_features *= 0 

        # features = np.concatenate([video_features, audio_features], axis=1)
        captions = self.vid_cap_dict[video_id_full]

        return video_id_full, torch.tensor(audio_features), torch.tensor(video_features), captions


class VideoCaptionsCollect:
    """
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]

    """

    def __init__(self):
        pass

    def __call__(self, batch):
        video_ids = [item[0] for item in batch]

        audio_features = [item[1] for item in batch]
        audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)

        visual_features = [item[2] for item in batch]
        visual_features = pad_sequence(visual_features, batch_first=True, padding_value=0)

        captions = [item[3] for item in batch]
        return video_ids, audio_features, visual_features, captions


def VideoDataset_to_VideoCaptionsLoader(videodataset, batch_size=32, num_workers=0, normalize=False, video_only=False):

    df = pd.DataFrame()
    df["video_id"] = videodataset.metadata["video_id"]
    df["caption"] = videodataset.metadata["caption"].apply(videodataset.vocab.apply_vocab)
    vid_captions_dict = df[["video_id", "caption"]].groupby("video_id")["caption"].apply(list).to_dict()

    videoCaptionsDataset = VideoCaptionsDataset(videodataset.root_dir, vid_captions_dict, normalize=normalize, video_only=video_only)

    loader = DataLoader(
        dataset=videoCaptionsDataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=VideoCaptionsCollect(),
    )

    return loader


class CustomCollate:
    """
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]

    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # features = [item[0].unsqueeze(0) for item in batch]
        # features = torch.cat(features, dim=0)
        features = [item[0] for item in batch]
        features = pad_sequence(features, batch_first=True, padding_value=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return features, captions


class CustomCollateAV:
    """
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]

    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        audio_features = [item[0] for item in batch]
        audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)

        video_features = [item[1] for item in batch]
        video_features = pad_sequence(video_features, batch_first=True, padding_value=0)

        captions = [item[2] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return audio_features, video_features, captions


def get_loader(
    root_dir,
    dataset="MSVD",
    split="train",
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    vocab_pkl=None,
    normalize=False,
    video_only=False,
):
    print('-'*50)
    print("Initializing loader:")
    print("Dataset:", dataset)
    print("Split:", split)
    print("Video_only ?:", video_only)
    print('-'*50)

    dataset = VideoCaptioningDataset(root_dir, dataset=dataset, split=split, vocab_pkl=vocab_pkl, normalize=normalize, video_only=video_only)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CustomCollateAV(pad_idx=pad_idx),
    )
    return loader, dataset


if __name__ == "__main__":
    ## one time setup
    # build_MSVD_vocab()

    dataset = "MSR-VTT"

    dataset_folder = os.path.join("datasets", dataset)
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")

    train_loader, train_dataset = get_loader(root_dir=dataset_folder, dataset=dataset, split="train", batch_size=128)
    val_loader, val_dataset = get_loader(root_dir=dataset_folder, dataset=dataset, split="val", batch_size=128, vocab_pkl=vocab_pkl)
    test_loader, test_dataset = get_loader(
        root_dir=dataset_folder, dataset=dataset, split="test", batch_size=1, shuffle=False, vocab_pkl=vocab_pkl
    )

    print(len(train_dataset.vocab))

    for loader in [train_loader, val_loader]:  # , val_loader, test_loader]:
        for idx, (audio_features, visual_features, captions) in enumerate(loader):
            # features, captions = features.cuda(), captions.cuda()

            captions_decoded = train_dataset.vocab.decode_indexes(captions[:, 0])

            # output, recons = model.decode(features, captions)
            print(idx, audio_features.shape, visual_features.shape, captions.shape)#, output.shape)
            # print("-" * 50)
            # print(idx)
            # print("\nAudio:")
            # print(audio_features.shape)
            # print("\nVisual")
            # print(visual_features.shape)
            # print("\nCaption:")
            # print(captions[:,0])
            # print(captions_decoded)
            # print("-" * 50)

            if idx == 5:
                break


    # val_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(val_loader.dataset, 1)

    # for loader in [val_vidCap_loader]:  # , val_loader, test_loader]:
    #     for idx, (video_id, audio_features, visual_features, captions) in enumerate(loader):
    #         # features, captions = features.cuda(), captions.cuda()

    #         # captions_decoded = train_dataset.vocab.decode_indexes(captions[0])

    #         # output, recons = model.decode(features, captions)
    #         # print(idx, features.shape, captions.shape, output.shape)
    #         print("-" * 50)
    #         print(idx, video_id)
    #         print("\nAudio:")
    #         print(audio_features)
    #         print("\nVisual")
    #         print(visual_features)
    #         print("\nCaption:")
    #         print(captions)
    #         # print(captions_decoded)
    #         print("-" * 50)

    #         # if idx > 1:
    #         #     audio_diff = audio_features - last_audio_features
    #         #     visual_diff = visual_features - last_visual_features

    #         #     print(
    #         #         f"Equal ? : {torch.eq(audio_features, last_audio_features).all()}, {torch.eq(visual_features, last_visual_features).all()}"
    #         #     )

    #             # print(audio_diff.sum())
    #             # print(visual_diff.sum())

    #         # last_audio_features = torch.clone(audio_features)
    #         # last_visual_features = torch.clone(visual_features)

    #         if idx == 5:
    #             break
