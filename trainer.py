import argparse
import gc
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from losses import (
    ModalityWiseReconstructionLoss,
    ModalityWiseReconstructionLossBuilder,
    ReconstructionLossBuilder,
    TotalReconstructionLoss,
)
from models import AVCaptioning
from get_loader import Vocabulary, get_loader, VideoDataset_to_VideoCaptionsLoader
from losses import ReconstructionLossBuilder, TotalReconstructionLoss, NLPScore
from models import AVCaptioning

import gc

DEBUG = False ## FIXME: Before training

class TrainerConfig:
    batch_size = 128

    epochs = 2 if DEBUG else 2 # FIXME
    lr = 1e-4
    weight_decay = 1e-5
    optimizer = optim.Adam
    gradient_clip_value = 5.0

    # lr_scheduler
    lr_decay_gamma = 0.5
    lr_decay_patience = 5

    ## Reconstructor Regularizer
    reg_lambda = 0.001  # 0.001
    audio_recon_lambda = 10
    visual_recon_lambda = 10

class Trainer:
    def __init__(self, checkpoint_name, display_freq=10):
        # assert checkpoint_name.endswith(".tar"), "The checkpoint file must have .tar extension"
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq

    def _load_checkpoint(self, model):
        if os.path.isfile(self.checkpoint_name):
            try:
                print(f"Resuming training from checkpoint: {self.checkpoint_name}")
                checkpoint = torch.load(self.checkpoint_name)

                model.decoder.load_state_dict(checkpoint["decoder"])
                if model.reconstructor and checkpoint["reconstructor"]:
                    model.reconstructor.load_state_dict(checkpoint["reconstructor"])

                self.history = checkpoint["history"]
            except:
                print(f"Error loading from checkpoint: {self.checkpoint_name}. \nUsing default parameters...")
        else:
            print(f"No checkpoint found, using default parameters...")
        return model

    def _save_checkpoint(self, epochs, model, config):
        checkpoint_base = os.path.dirname(self.checkpoint_name)
        if not os.path.exists(checkpoint_base):
            os.makedirs(checkpoint_base)

        torch.save(
            {
                "epoch": epochs,
                "decoder": model.decoder.state_dict(),
                "reconstructor": model.reconstructor.state_dict() if model.reconstructor else None,
                # 'config': cls_to_dict(config),
                "history": self.history,
            },
            self.checkpoint_name,
        )

    def fit(self, model, train_loader, val_loader, test_loader, device, train_config):
        self.device = device

        # Training utils
        self.optimizer = train_config.optimizer(
            model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay, amsgrad=True
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=train_config.lr_decay_gamma,
            patience=train_config.lr_decay_patience,
            verbose=True,
        )
        self.gradient_clip_value = train_config.gradient_clip_value

        self.reg_lambda = train_config.reg_lambda
        self.audio_recon_lambda = train_config.audio_recon_lambda
        self.visual_recon_lambda = train_config.visual_recon_lambda

        self.RecLoss = ModalityWiseReconstructionLossBuilder(
            reg_lambda=self.reg_lambda,
            audio_recon_lambda=self.audio_recon_lambda,
            visual_recon_lambda=self.visual_recon_lambda,
            rec_type=model.reconstructor_type,
        )

        self.history = {"train_loss": [], "val_loss": [], "val_score": [], "test_loss": [], "test_score": []}
        self.previous_epochs = 0
        self.best_loss = 1e6

        ## VideoCaptionsDataloader for Evaluation
        val_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(val_loader.dataset, 1)#train_config.batch_size)
        test_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(test_loader.dataset, 1) #train_config.batch_size)

        # Start training
        for epoch in range(self.previous_epochs + 1, train_config.epochs + 1):
            print(f"\nEpoch {epoch}/{train_config.epochs}:")

            # train_loss = self.train(model, train_loader)
            # val_loss = self.test(model, val_loader)
            val_score = self.eval(model, val_vidCap_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_score"].append(val_score)

            # Scheduler step, check plateau
            self.lr_scheduler.step(val_loss["total"])

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if val_loss["total"] < self.best_loss:
                print(f"Validation loss improved from {self.best_loss} to {val_loss['total']}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")

                self.best_loss = val_loss["total"]
                self._save_checkpoint(epoch, model, {})  # FIXME: empty config

        test_loss = self.test(model, test_loader)
        test_score = self.eval(model, test_vidCap_loader)

        self.history["test_loss"].append(test_loss)
        self.history["test_score"].append(test_score)

        return self.history

    def train(self, model, dataloader):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        audio_reconstruction_loss = 0.0
        visual_reconstruction_loss = 0.0

        model.train()

        with tqdm(dataloader, desc="TRAIN") as progress:
            for i, (audio_features, visual_features, captions) in enumerate(progress):
                self.optimizer.zero_grad()
                audio_features, visual_features, captions = (
                    audio_features.to(self.device),
                    visual_features.to(self.device),
                    captions.to(self.device),
                )

                outputs, audio_recons, visual_recons = model(audio_features, visual_features, captions)

                loss, ce, e, a_recon, v_recon = self.RecLoss(
                    outputs,
                    captions,
                    audio_features,  # Audio features [1000-1128]
                    audio_recons,
                    visual_features,  # Visual features [0-1000]
                    visual_recons,
                )
                loss.mean().backward()

                if self.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.gradient_clip_value)

                self.optimizer.step()

                total_loss += loss.mean().item()
                cross_entropy_loss += ce.mean().item()
                entropy_loss += e.mean().item()
                audio_reconstruction_loss += a_recon.mean().item()
                visual_reconstruction_loss += v_recon.mean().item()

                if i % self.display_freq == 0:
                    progress.set_postfix(
                        {
                            "total": float(total_loss / (i + 1)),
                            "ce": float(cross_entropy_loss / (i + 1)),
                            "e": float(entropy_loss / (i + 1)),
                            "a_recon": float(audio_reconstruction_loss / (i + 1)),
                            "v_recon": float(visual_reconstruction_loss / (i + 1)),
                        }
                    )

        # Garbage collector (fix memory allocation problems ? )
        gc.collect()

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "a_recon": audio_reconstruction_loss / len(dataloader),
            "v_recon": visual_reconstruction_loss / len(dataloader),
        }

    def test(self, model, dataloader):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        audio_reconstruction_loss = 0.0
        visual_reconstruction_loss = 0.0

        model.eval()

        with torch.no_grad():
            with tqdm(dataloader, desc="TEST ") as progress:
                for i, (audio_features, visual_features, captions) in enumerate(progress):
                    audio_features, visual_features, captions = (
                        audio_features.to(self.device),
                        visual_features.to(self.device),
                        captions.to(self.device),
                    )

                    outputs, audio_recons, visual_recons = model(audio_features, visual_features, captions)
                    loss, ce, e, a_recon, v_recon = self.RecLoss(
                        outputs,
                        captions,
                        audio_features,  # Audio features [1000-1128]
                        audio_recons,
                        visual_features,  # Visual features [0-1000]
                        visual_recons,
                    )
                    total_loss += loss.mean().item()
                    cross_entropy_loss += ce.mean().item()
                    entropy_loss += e.mean().item()
                    audio_reconstruction_loss += a_recon.mean().item()
                    visual_reconstruction_loss += v_recon.mean().item()

                    if i % self.display_freq == 0:
                        progress.set_postfix(
                            {
                                "total": float(total_loss / (i + 1)),
                                "ce": float(cross_entropy_loss / (i + 1)),
                                "e": float(entropy_loss / (i + 1)),
                                "a_recon": float(audio_reconstruction_loss / (i + 1)),
                                "v_recon": float(visual_reconstruction_loss / (i + 1)),
                            }
                        )

        # Garbage collector (fix memory allocation problems ? )
        gc.collect()

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "a_recon": audio_reconstruction_loss / len(dataloader),
            "v_recon": visual_reconstruction_loss / len(dataloader),
        }

    def eval(self, model, videoCaptions_dataloader):
        model.eval()
        vid_GT = {}
        vid_gen = {}
        with tqdm(videoCaptions_dataloader, desc="EVAL ") as progress:
            for i, (vid_ids, audio_features, visual_features, captions) in enumerate(progress):
                audio_features, visual_features = (
                    audio_features.to(self.device),
                    visual_features.to(self.device)
                )
                generated_captions = model.predict(audio_features, visual_features, max_caption_len=30, beam_alpha=0, beam_width=5)
                
                # generated_captions = generated_captions.detach().cpu()
                
                # print(captions)
                # print(generated_captions)

                vid_GT.update({k:v for k, v in zip(vid_ids, captions)})
                vid_gen.update({k:[v] for k, v in zip(vid_ids, generated_captions)})

                # FIXME: only for debug
                if i >= 4:
                    break

        scores = NLPScore(vid_GT, vid_gen) 
        print(scores)
        return scores

if __name__ == "__main__":
    from models import FeaturesCaptioning

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_folder = os.path.join("datasets", "MSVD")
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    vocab = Vocabulary.load(vocab_pkl)

    train_config = TrainerConfig()


    train_loader, train_dataset = get_loader(
        root_dir=dataset_folder,
        split="tiny" if DEBUG else "train",
        batch_size=train_config.batch_size,
        vocab_pkl=vocab_pkl,
    )
    val_loader, _ = get_loader(
        root_dir=dataset_folder,
        split="tiny" if DEBUG else "val",
        batch_size=train_config.batch_size,
        vocab_pkl=vocab_pkl,
    )
    test_loader, _ = get_loader(
        root_dir=dataset_folder,
        split="tiny" if DEBUG else "test",
        batch_size=train_config.batch_size,
        vocab_pkl=vocab_pkl,
    )

    CHECKPOINTS_DIR = os.path.join("checkpoints")

    # experiments = [

    # ]

    model = AVCaptioning(
        vocab=vocab,
        teacher_forcing_ratio=0.5,
        no_reconstructor=True,
        device=device,
    )
    model.to(device)

    print("Start training")
    tr = Trainer(checkpoint_name=os.path.join(CHECKPOINTS_DIR, "test.ckpt"))
    history = tr.fit(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        train_config,
    )
