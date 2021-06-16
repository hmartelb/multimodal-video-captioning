import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from get_loader import Vocabulary, get_loader
from losses import (EntropyLoss, GlobalReconstructionLoss,
                    LocalReconstructionLoss, TotalReconstructionLoss)


class Trainer:
    def __init__(self, checkpoint_name, display_freq=10):
        # self.train_data = train_data
        # self.val_data = val_data
        # assert checkpoint_name.endswith(".tar"), "The checkpoint file must have .tar extension"
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq

    def _load_checkpoint(self, model):
        if os.path.isfile(self.checkpoint_name):
            print(f"Resuming training from checkpoint: {self.checkpoint_name}")
            checkpoint = torch.load(self.checkpoint_name)
            model.decoder.load_state_dict(checkpoint["decoder"])
            if model.reconstructor and checkpoint["reconstructor"]:
                model.reconstructor.load_state_dict(checkpoint["reconstructor"])

            self.history = checkpoint["history"]
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

    def fit(
        self,
        model,
        train_loader,
        val_loader,
        device,
        epochs=10,
        batch_size=16,
        lr=0.0001,
        weight_decay=1e-5,
        optimizer=optim.Adam,
        lr_decay_gamma=0,  # FIXME
        lr_decay_patience=4,  # FIXME
        gradient_clip_value=0,
    ):
        # Get the device placement and make data loaders
        self.device = device
        # kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
        # self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, **kwargs)
        # self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, **kwargs)

        # Get dataloaders and vocab
        # dataset_folder = os.path.join("datasets", "MSVD")
        # vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
        # self.train_loader, train_dataset = get_loader(
        #     root_dir=dataset_folder,
        #     split="train",
        #     batch_size=batch_size,
        # )
        # self.val_loader, _ = get_loader(
        #     root_dir=dataset_folder,
        #     split="val",
        #     batch_size=batch_size,
        #     vocab_pkl=vocab_pkl,
        # )
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.vocab = train_dataset.vocab
        # print(f"Data loaders ready: {dataset_folder}")
        # print(f"Vocab size: {len(self.vocab)}")

        # Training utils
        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_decay_gamma,
            patience=lr_decay_patience,
            verbose=True,
        )
        self.gradient_clip_value = gradient_clip_value
        self.history = {"train_loss": [], "test_loss": []}

        self.previous_epochs = 0
        self.best_loss = None

        # Start training
        for epoch in range(self.previous_epochs + 1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}:")

            train_loss = self.train(model)
            val_loss = self.test(model)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Scheduler step, check plateau
            self.lr_scheduler.step(val_loss["total"])

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if self.best_loss is None or (val_loss["total"] < self.best_loss):

                print(f"Validation loss improved from {best_loss} to {val_loss['total']}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")

                self.best_loss = val_loss["total"]
                self._save_checkpoint(epoch, model, {})  # FIXME: empty config

        return self.history

    def train(self, model):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        reconstruction_loss = 0.0

        model.train()

        with tqdm(self.train_loader) as progress:
            for i, (features, captions) in enumerate(progress):
                self.optimizer.zero_grad()
                features, captions = features.to(self.device), captions.to(self.device)               

                output, features_recons = model.decode(features, captions, max_caption_len=captions.shape[0])
                loss, ce, e, recon = TotalReconstructionLoss(
                    output,
                    captions,
                    features,
                    features_recons,
                    reg_lambda=0,
                    recon_lambda=0,
                )
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
        return {
            "total": total_loss / len(self.train_loader),
            "ce": cross_entropy_loss / len(self.train_loader),
            "e": entropy_loss / len(self.train_loader),
            "recon": reconstruction_loss / len(self.train_loader),
        }

    def test(self, model):
        total_loss = 0.0
        cross_entropy_loss = 0.0
        entropy_loss = 0.0
        reconstruction_loss = 0.0

        model.eval()

        with torch.no_grad():
            with tqdm(self.val_loader) as progress:
                for i, (features, captions) in enumerate(progress):
                    features, captions = features.to(self.device), captions.to(self.device)

                    output, features_recons = model.decode(features, max_caption_len=captions.shape[0])
                    loss, ce, e, recon = TotalReconstructionLoss(
                        output,
                        captions,
                        features,
                        features_recons,
                        reg_lambda=0,
                        recon_lambda=0,
                    )

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
        return {
            "total": total_loss / len(self.val_loader),
            "ce": cross_entropy_loss / len(self.val_loader),
            "e": entropy_loss / len(self.val_loader),
            "recon": reconstruction_loss / len(self.val_loader),
        }


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_folder = os.path.join("datasets", "MSVD")
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    vocab = Vocabulary.load(vocab_pkl)

    train_loader, train_dataset = get_loader(
        root_dir=dataset_folder,
        split="train",
        batch_size=batch_size,
        vocab_pkl=vocab_pkl,
    )
    val_loader, _ = get_loader(
        root_dir=dataset_folder,
        split="val",
        batch_size=batch_size,
        vocab_pkl=vocab_pkl,
    )

    from models import Decoder

    model = Decoder(
        output_size=3056, # FIXME: Use vocab size. For some reason, vocab_pkl is 3201 
        attn_size=128,
        max_caption_len=30,
    )
    model = model.cuda()

    print("Start training")
    tr = Trainer(checkpoint_name=os.path.join("checkpoints", "test.ckpt"))
    tr.fit(
        model,
        train_loader,
        val_loader,
        device,
        epochs=1,
        batch_size=128,
    )
