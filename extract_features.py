import os

import cv2
import moviepy.editor as mp
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

from models import AudioEncoder, VisualEncoder


def make_path(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


class FeatureExtractor:
    def __init__(self):
        self.audio_extractor = AudioEncoder(model="vggish", trainable=False)
        self.image_extractor = VisualEncoder(model="inception_v3", trainable=False, normalize=True)

        self.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def extract(self, filename, fps=1, to_numpy=False):
        video_f, audio_f = self.get_video_features(filename, fps), self.get_audio_features(filename)
        if to_numpy:
            video_f = video_f.detach().cpu().numpy()
            audio_f = audio_f.detach().cpu().numpy()
        return video_f, audio_f

    def get_audio_features(self, filename):
        #
        # FIXME: This is not very elegant, but it works...
        #
        # 1) Extract and export the audio track
        # 2) Load with torchaudio
        _, ext = os.path.splitext(filename)
        audio_filename = filename.replace("videos", "audios").replace(ext, ".wav")
        # print(audio_filename)

        if not os.path.isfile(audio_filename):
            # Step 1
            clip = mp.VideoFileClip(filename)
            if clip.audio:
                clip.audio.write_audiofile(audio_filename, codec="pcm_s16le", verbose=False, logger=None)
            else:
                audio = AudioSegment.silent(duration=1000 * clip.duration, frame_rate=16000)
                audio.export(audio_filename, format="wav")

        # Step 2
        audio, sr = torchaudio.load(audio_filename)

        return self.audio_extractor.from_tensor(audio, sr)

    def get_video_features(self, filename, fps=1):
        frames = self._load_frames(filename, fps=fps)
        frames = torch.Tensor(frames).to(self.device)
        return self.image_extractor(frames)

    def set_device(self, device):
        self.device = device
        self.audio_extractor = self.audio_extractor.to(device)
        self.image_extractor = self.image_extractor.to(device)

    def cuda(self):
        self.set_device(torch.device("cuda"))

    def _load_frames(
        self,
        filename,
        frame_size=(299,299),
        take_n=0,
        fps=None,
        fmt="channels_first",
    ):
        frames = []
        cap = cv2.VideoCapture(filename)
        frame_rate = np.round(cap.get(cv2.CAP_PROP_FPS))

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
                frames.append(frame)
            else:
                break

        frames = np.array(frames)

        # Downsample the video to the specified fps
        if fps and fps < frame_rate:
            total_frames = frames.shape[0]
            take_n = int(frames.shape[0] // frame_rate * fps)

        if take_n > 0:
            total_frames = frames.shape[0]
            frame_idx = np.arange(0, total_frames, total_frames // take_n)
            frames = frames[frame_idx, :, :, :]

        if fmt == "channels_first":  # [n_frames, n_channels, w, h]
            frames = np.transpose(frames, (0, 3, 1, 2))

        return frames


if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument("--gpu", default="-1")
    args = ap.parse_args()

    # Select which GPU to use and enable mixed precision
    print("Using GPU: " + args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    assert os.path.isdir(args.dataset), "The specified directory does not exist"

    DATASET_DIR = args.dataset
    VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")
    AUDIOS_DIR = os.path.join(DATASET_DIR, "audios")

    FEATURES_DIR = os.path.join(DATASET_DIR, "features")
    VIDEO_FEATURES_DIR = os.path.join(FEATURES_DIR, "video")
    AUDIO_FEATURES_DIR = os.path.join(FEATURES_DIR, "audio")

    make_path(VIDEO_FEATURES_DIR)
    make_path(AUDIO_FEATURES_DIR)

    fe = FeatureExtractor()

    failures = []
    with tqdm(os.listdir(VIDEOS_DIR)) as progress:
        for f in progress:
            name, ext = os.path.splitext(f)
            progress.set_postfix({
                "Processing file": f"{name}{ext}",
                "Failures": len(failures)
            })
            video_name = os.path.join(VIDEOS_DIR, f)
            audio_name = os.path.join(AUDIOS_DIR, f.replace(ext, '.wav'))

            video_features_name = os.path.join(VIDEO_FEATURES_DIR, name)
            audio_features_name = os.path.join(AUDIO_FEATURES_DIR, name)

            data_exists = (os.path.isfile(video_name) and os.path.isfile(audio_name))
            features_exist = (os.path.isfile(video_features_name) and os.path.isfile(audio_features_name))

            if data_exists and not features_exist:
                # try:
                video_f, audio_f = fe.extract(os.path.join(VIDEOS_DIR, f), to_numpy=True)
                
                np.save(video_features_name, video_f)
                np.save(audio_features_name, audio_f)

                # except:
                #     failures.append(f"{name}{ext}")

    print(failures)