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
        pickle.dump(self, open(path, 'wb+'))
        print(f"Vocab saved: {path}")

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

    def apply_vocab(self, sentence):
        '''
        convert token not in vocab to "<UNK>"
        '''
        tokenized_text = self.tokenizer_eng(sentence)
        tokens = [token if token in self.stoi else "<UNK>" for token in tokenized_text]
        return ' '.join(tokens)

    def decode_indexes(self, indexes):
        words = []
        EOS_idx = 2
        for idx in indexes:
            if type(idx) is torch.Tensor:
                idx = idx.item()
            if idx == EOS_idx:
                break
            words.append(self.itos[idx])
        sentence = ' '.join(words)
        return sentence


def build_MSVD_vocab():
    dataset_folder = os.path.join("datasets", "MSVD")
    train_captions_file = os.path.join(dataset_folder, "metadata", "train.csv")
    val_captions_file = os.path.join(dataset_folder, "metadata", "val.csv")
    train_captions = pd.read_csv(train_captions_file)["Description"].tolist()
    val_captions = pd.read_csv(val_captions_file)["Description"].tolist()
    vocab_path = os.path.join(dataset_folder, "metadata", "vocab.pkl")

    Vocabulary.prebuild(train_captions + val_captions, vocab_path)


class MSVD_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        freq_threshold=5,
        vocab_pkl=None,
    ):
        self.root_dir = root_dir

        assert os.path.isdir(root_dir), "The dataset root directory does not exist"
        assert os.path.isdir(os.path.join(root_dir, "metadata")), "The dataset metadata directory does not exist"
        assert os.path.isdir(os.path.join(root_dir, "features")), "The dataset features directory does not exist"
        assert split in ["train", "val", "test", "tiny"], "Wrong split specified, must be one of ['train', 'val', 'test']"

        self.captions_file = os.path.join(root_dir, "metadata", f"{split}.csv")
        assert os.path.isfile(self.captions_file), f"The captions file cannot be found {self.captions_file}"
        if vocab_pkl is not None:
            assert os.path.isfile(vocab_pkl), f"The vocab file cannot be found {vocab_pkl}"

        self.metadata = pd.read_csv(self.captions_file)

        # Initialize vocabulary and build vocab
        if vocab_pkl is None:
            print("Building Vocab")
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.metadata["Description"].tolist())
        else:
            print(f"Loading Vocab: {vocab_pkl} ")
            self.vocab = Vocabulary.load(vocab_pkl)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        video_id, start, end, caption = row["VideoID"], row["Start"], row["End"], row["Description"]
        caption_tokens = [self.vocab.stoi["<SOS>"]]
        caption_tokens += self.vocab.numericalize(caption)
        caption_tokens.append(self.vocab.stoi["<EOS>"])

        video_features_file = os.path.join(self.root_dir, "features", "video", f"{video_id}_{start}_{end}.npy")
        audio_features_file = os.path.join(self.root_dir, "features", "audio", f"{video_id}_{start}_{end}.npy")

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
        video_features /= np.sum(video_features, axis=1, keepdims=True)
        audio_features /= np.sum(audio_features, axis=1, keepdims=True)

        return torch.tensor(audio_features), torch.tensor(video_features), torch.tensor(caption_tokens)

        # features = np.concatenate([video_features, audio_features], axis=1)
        # return torch.tensor(features), torch.tensor(caption_tokens)

class VideoCaptionsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        vid_cap_dict,
    ):
        '''
        vid_cap_dict: dict({ vid: [captions] })
        '''
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
        video_features /= np.sum(video_features, axis=1, keepdims=True)
        audio_features /= np.sum(audio_features, axis=1, keepdims=True)

        # features = np.concatenate([video_features, audio_features], axis=1)
        captions = self.vid_cap_dict[video_id_full]

        return video_id_full, torch.tensor(audio_features), torch.tensor(video_features), captions

class VideoCaptionsCollect:
    '''
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]
    
    '''
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

def VideoDataset_to_VideoCaptionsLoader(videodataset, batch_size=32, num_workers=0):

    full_video_id = lambda x: f"{x['VideoID']}_{x['Start']}_{x['End']}"
    df = pd.DataFrame()
    df["FullVideoID"] = videodataset.metadata.apply(full_video_id, axis=1)
    df["Caption"] = videodataset.metadata["Description"].apply(videodataset.vocab.apply_vocab)
    vid_captions_dict = df[['FullVideoID', 'Caption']].groupby('FullVideoID')['Caption'].apply(list).to_dict()

    videoCaptionsDataset = VideoCaptionsDataset(videodataset.root_dir, vid_captions_dict)

    loader = DataLoader(
        dataset=videoCaptionsDataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=VideoCaptionsCollect(),
    )

    return loader

class CustomCollate:
    '''
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]
    
    '''
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
    '''
    return batch data (features, captions) in the shape of:
    features: [batchsize, length, feat_dim]
    captions: [length, batchsize]
    
    '''
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
    split="train",
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    vocab_pkl=None,
):
    dataset = MSVD_Dataset(root_dir, split=split, vocab_pkl=vocab_pkl)
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

    dataset_folder = os.path.join("datasets", "MSVD")
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    train_loader, train_dataset = get_loader(root_dir=dataset_folder, split="train", batch_size=32)
    val_loader, val_dataset = get_loader(root_dir=dataset_folder, split="val", batch_size=16, vocab_pkl=vocab_pkl)
    test_loader, test_dataset = get_loader(root_dir=dataset_folder, split="test", batch_size=1, shuffle=False, vocab_pkl=vocab_pkl)

    print(len(train_dataset.vocab))

    from models import Decoder
    model = Decoder(
        output_size=3056,
        attn_size=128,
        max_caption_len=18,
    )
    model = model.cuda()

    for loader in [train_loader]:#, val_loader, test_loader]:
        for idx, (features, captions) in enumerate(loader):
            features, captions = features.cuda(), captions.cuda()

            output, recons = model.decode(features, captions)
            print(idx, features.shape, captions.shape, output.shape)
            if idx == 50:
                break
