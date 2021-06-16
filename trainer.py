import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from get_loader import Vocabulary, get_loader
from losses import ReconstructionLossBuilder, TotalReconstructionLoss
from models import AVCaptioning

import gc


class TrainerConfig:
    batch_size = 8

    epochs = 50
    lr = 5e-5
    weight_decay = 1e-5
    optimizer = optim.Adam
    gradient_clip_value = 5.0

    # lr_scheduler
    lr_decay_gamma = 0.5  
    lr_decay_patience = 5  

    ## Reconstructor Regularizer
    reg_lambda = 0  # 0.001
    recon_lambda = 0


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
                # "decoder": model.state_dict(),  ## temp FIX
                "decoder": model.decoder.state_dict(),
                "reconstructor": model.reconstructor.state_dict() if model.reconstructor else None,
                # 'config': cls_to_dict(config),
                "history": self.history,
            },
            self.checkpoint_name,
        )

    def fit(self, model, train_loader, val_loader, device, train_config):
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
        self.recon_lambda = train_config.recon_lambda
        self.history = {"train_loss": [], "val_loss": [], "test_loss": []}

        self.RecLoss = ReconstructionLossBuilder(
            reg_lambda=self.reg_lambda, recon_lambda=self.recon_lambda, reconstruction_type=model.reconstructor_type
        )

        self.previous_epochs = 0
        self.best_loss = 1e6

        # Start training
        for epoch in range(self.previous_epochs + 1, train_config.epochs + 1):
            print(f"\nEpoch {epoch}/{train_config.epochs}:")

            train_loss = self.train(model, train_loader)
            val_loss = self.test(model, val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Scheduler step, check plateau
            self.lr_scheduler.step(val_loss["total"])

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if val_loss["total"] < self.best_loss:
                print(f"Validation loss improved from {self.best_loss} to {val_loss['total']}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")

                self.best_loss = val_loss["total"]
                self._save_checkpoint(epoch, model, {})  # FIXME: empty config

        return self.history

    def train(self, model, dataloader):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        reconstruction_loss = 0.0

        model.train()

        with tqdm(dataloader, desc="TRAIN") as progress:
            for i, (features, captions) in enumerate(progress):
                self.optimizer.zero_grad()
                features, captions = features.to(self.device), captions.to(self.device)

                outputs, features_recons = model(features, captions)

                loss, ce, e, recon = self.RecLoss(
                    outputs,
                    captions,
                    features,
                    features_recons,
                )
                # loss, ce, e, recon = TotalReconstructionLoss(
                #     outputs,
                #     captions,
                #     features,
                #     features_recons,
                #     reg_lambda=0,
                #     recon_lambda=0,
                #     reconstruction_type=model.reconstructor_type,
                # )
                loss.mean().backward()

                if self.gradient_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.gradient_clip_value)

                self.optimizer.step()

                total_loss += loss.mean().item()
                cross_entropy_loss += ce.mean().item()
                entropy_loss += e.mean().item()
                reconstruction_loss += recon.mean().item()

                if i % self.display_freq == 0:
                    progress.set_postfix(
                        {
                            "total": float(total_loss / (i + 1)),
                            "ce": float(cross_entropy_loss / (i + 1)),
                            "e": float(entropy_loss / (i + 1)),
                            "recon": float(reconstruction_loss / (i + 1)),
                        }
                    )

        # Garbage collector (fix memory allocation problems ? )
        gc.collect()

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "recon": reconstruction_loss / len(dataloader),
        }

    def test(self, model, dataloader):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        reconstruction_loss = 0.0

        model.eval()

        with torch.no_grad():
            with tqdm(dataloader, desc="TEST ") as progress:
                for i, (features, captions) in enumerate(progress):
                    features, captions = features.to(self.device), captions.to(self.device)

                    outputs, features_recons = model(features, captions)

                    loss, ce, e, recon = self.RecLoss(
                        outputs,
                        captions,
                        features,
                        features_recons,
                    )
                    # loss, ce, e, recon = TotalReconstructionLoss(
                    #     outputs,
                    #     captions,
                    #     features,
                    #     features_recons,
                    #     reg_lambda=0,
                    #     recon_lambda=0,
                    #     reconstruction_type=model.reconstructor_type,
                    # )

                    total_loss += loss.mean().item()
                    cross_entropy_loss += ce.mean().item()
                    entropy_loss += e.mean().item()
                    reconstruction_loss += recon.mean().item()

                    if i % self.display_freq == 0:
                        progress.set_postfix(
                            {
                                "total": float(total_loss / (i + 1)),
                                "ce": float(cross_entropy_loss / (i + 1)),
                                "e": float(entropy_loss / (i + 1)),
                                "recon": float(reconstruction_loss / (i + 1)),
                            }
                        )

        # Garbage collector (fix memory allocation problems ? )
        gc.collect()

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "recon": reconstruction_loss / len(dataloader),
        }


if __name__ == "__main__":
    from models import FeaturesCaptioning

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_folder = os.path.join("datasets", "MSVD")
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    vocab = Vocabulary.load(vocab_pkl)

    train_config = TrainerConfig()

    DEBUG = False

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

    model = AVCaptioning(
        vocab_size=len(vocab),
        teacher_forcing_ratio=0.5,
        no_reconstructor=False,
        device=device,
    )
    model.to(device)

    print("Start training")
    tr = Trainer(checkpoint_name=os.path.join("checkpoints", "test.ckpt"))
    tr.fit(
        model,
        train_loader,
        val_loader,
        device,
        train_config,
    )
