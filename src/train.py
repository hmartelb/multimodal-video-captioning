import argparse
import gc
import os

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from get_loader import VideoDataset_to_VideoCaptionsLoader, Vocabulary, get_loader
from losses import ModalityWiseReconstructionLossBuilder, NLPScore
from models import AVCaptioning, AVCaptioningDual


class TrainerConfig:
    batch_size = 128

    epochs = 50
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
    def __init__(self, checkpoint_name, log_dir="logs", display_freq=10, eval_freq=10):
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq
        self.eval_freq = eval_freq

        self.summary_writer = SummaryWriter(log_dir)

    def _load_checkpoint(self, model):
        if os.path.isfile(self.checkpoint_name):
            try:
                print(f"Resuming training from checkpoint: {self.checkpoint_name}")
                checkpoint = torch.load(self.checkpoint_name)

                model.v_decoder.load_state_dict(checkpoint["v_decoder"])
                if model.v_reconstructor and checkpoint["v_reconstructor"]:
                    model.v_reconstructor.load_state_dict(checkpoint["v_reconstructor"])

                model.a_decoder.load_state_dict(checkpoint["a_decoder"])
                if model.a_reconstructor and checkpoint["a_reconstructor"]:
                    model.a_reconstructor.load_state_dict(checkpoint["a_reconstructor"])

                self.previous_epochs = checkpoint["epoch"]
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
                "v_decoder": model.v_decoder.state_dict(),
                "a_decoder": model.a_decoder.state_dict(),
                "v_reconstructor": model.v_reconstructor.state_dict() if model.v_reconstructor else None,
                "a_reconstructor": model.a_reconstructor.state_dict() if model.a_reconstructor else None,
                "history": self.history,
            },
            self.checkpoint_name,
        )

    def fit(self, model, train_loader, val_loader, test_loader, device, train_config, ):
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
            min_lr=1e-7,
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

        self.history = {
            "train_loss": [],
            "train_score": [],
            "val_loss": [],
            "val_score": [],
            "test_loss": [],
            "test_score": [],
        }
        self.previous_epochs = 0
        self.best_loss = 1e6
        self.best_CIDEr = 0

        model = self._load_checkpoint(model)
        model.to(self.device)

        # VideoCaptionsDataloader for Evaluation
        train_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(
            train_loader.dataset, train_config.batch_size, video_only=train_loader.dataset.video_only
        )
        val_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(
            val_loader.dataset, train_config.batch_size, video_only=val_loader.dataset.video_only
        )
        test_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(
            test_loader.dataset, train_config.batch_size, video_only=test_loader.dataset.video_only
        )

        # Start training
        for epoch in range(self.previous_epochs + 1, train_config.epochs + 1):
            print(f"\nEpoch {epoch}/{train_config.epochs}:")

            train_loss = self.train(model, train_loader, epoch)
            val_loss = self.test(model, val_loader, "val", epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % self.eval_freq == 0 or epoch == train_config.epochs:
                train_score, _, _ = self.eval(model, train_vidCap_loader, "train", epoch)
                val_score, _, _ = self.eval(model, val_vidCap_loader, "val", epoch)
                self.history["train_score"].append(train_score)
                self.history["val_score"].append(val_score)

                # Scheduler step, check plateau
                self.lr_scheduler.step(val_score["CIDEr"])

                # Save checkpoint based on EVAL METRIC
                if val_score["CIDEr"] > self.best_CIDEr:
                    print(f"CIDEr improved from {self.best_CIDEr} to {val_score['CIDEr']}.")
                    print(f"Saving checkpoint to: {self.checkpoint_name}")

                    self.best_CIDEr = val_score["CIDEr"]
                    self._save_checkpoint(epoch, model, {})
                    torch.save(model, self.checkpoint_name.replace(".ckpt", "_best.pt"))

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if val_loss["total"] < self.best_loss:
                print(f"Validation loss improved from {self.best_loss} to {val_loss['total']}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")

                self.best_loss = val_loss["total"]
                self._save_checkpoint(epoch, model, {})  # FIXME: empty config
                torch.save(model, self.checkpoint_name.replace(".ckpt", "_best.pt"))

        torch.save(model, self.checkpoint_name.replace(".ckpt", "_last.pt"))
        return self.history

    def train(self, model, dataloader, epoch):
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
                    outputs, captions, audio_features, audio_recons, visual_features, visual_recons,
                )
                loss.mean().backward()

                step_no = epoch * len(dataloader) + i
                self.summary_writer.add_scalar("train/loss", loss.mean().item(), step_no)
                self.summary_writer.add_scalar("train/loss/ce", ce.mean().item(), step_no)
                self.summary_writer.add_scalar("train/loss/e", e.mean().item(), step_no)
                self.summary_writer.add_scalar("train/loss/a_recon", a_recon.mean().item(), step_no)
                self.summary_writer.add_scalar("train/loss/v_recon", v_recon.mean().item(), step_no)

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

        self.summary_writer.add_scalar("train_epoch/loss", total_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar("train_epoch/loss/ce", cross_entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar("train_epoch/loss/e", entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar("train_epoch/loss/a_recon", audio_reconstruction_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar("train_epoch/loss/v_recon", visual_reconstruction_loss / len(dataloader), epoch)

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "a_recon": audio_reconstruction_loss / len(dataloader),
            "v_recon": visual_reconstruction_loss / len(dataloader),
        }

    def test(self, model, dataloader, training_phase, epoch):
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

                    outputs, audio_recons, visual_recons = model(
                        audio_features, visual_features, captions, teacher_forcing_ratio=0
                    )
                    loss, ce, e, a_recon, v_recon = self.RecLoss(
                        outputs, captions, audio_features, audio_recons, visual_features, visual_recons,
                    )

                    step_no = epoch * len(dataloader) + i
                    self.summary_writer.add_scalar(f"{training_phase}/loss", loss.mean().item(), step_no)
                    self.summary_writer.add_scalar(f"{training_phase}/loss/ce", ce.mean().item(), step_no)
                    self.summary_writer.add_scalar(f"{training_phase}/loss/e", e.mean().item(), step_no)
                    self.summary_writer.add_scalar(f"{training_phase}/loss/a_recon", a_recon.mean().item(), step_no)
                    self.summary_writer.add_scalar(f"{training_phase}/loss/v_recon", v_recon.mean().item(), step_no)

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

        self.summary_writer.add_scalar(f"{training_phase}_epoch/loss", total_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f"{training_phase}_epoch/loss/ce", cross_entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f"{training_phase}_epoch/loss/e", entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(
            f"{training_phase}_epoch/loss/a_recon", audio_reconstruction_loss / len(dataloader), epoch
        )
        self.summary_writer.add_scalar(
            f"{training_phase}_epoch/loss/v_recon", visual_reconstruction_loss / len(dataloader), epoch
        )

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "a_recon": audio_reconstruction_loss / len(dataloader),
            "v_recon": visual_reconstruction_loss / len(dataloader),
        }

    def eval(self, model, videoCaptions_dataloader, training_phase, epoch, mode="direct", get_scores=True):
        model.eval()
        vid_GT = {}
        vid_gen = {}
        with tqdm(videoCaptions_dataloader, desc="EVAL ") as progress:
            for i, (vid_ids, audio_features, visual_features, captions) in enumerate(progress):
                audio_features, visual_features = (audio_features.to(self.device), visual_features.to(self.device))
                generated_captions = model.predict(
                    audio_features, visual_features, max_caption_len=30, mode=mode, beam_alpha=0, beam_width=5
                )

                vid_GT.update({k: v for k, v in zip(vid_ids, captions)})
                vid_gen.update({k: [v] for k, v in zip(vid_ids, generated_captions)})

            print("\nExample captions: key >> [generated] (ground_truth)")
            for i, key in enumerate(vid_GT):
                print(f"{key} >> [{vid_gen[key][0]}] ({vid_GT[key][0]})")
                if i >= 10:
                    break
            print()

        scores = None
        if get_scores:
            scores = NLPScore(vid_GT, vid_gen)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/Bleu_1", scores["Bleu_1"], epoch)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/Bleu_2", scores["Bleu_2"], epoch)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/Bleu_3", scores["Bleu_3"], epoch)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/Bleu_4", scores["Bleu_4"], epoch)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/ROUGE_L", scores["ROUGE_L"], epoch)
            self.summary_writer.add_scalar(f"{training_phase}/score/{mode}/CIDEr", scores["CIDEr"], epoch)
            print(scores)

        return scores, vid_GT, vid_gen


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="-1")
    parser.add_argument("--dataset", default="MSVD", choices=["MSVD", "MSR-VTT"])
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset_folder = os.path.join("datasets", dataset)
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    vocab = Vocabulary.load(vocab_pkl)

    CHECKPOINTS_DIR = os.path.join("checkpoints", dataset)

    dual = True

    v_v_global_local_exp = [
        # # NO reconstructor
        # {
        #     "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "none"},
        #     "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
        #     "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0},
        #     "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_only",
        #     "log_dir": os.path.join("logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_only"),
        #     "video_only": True,
        # },
        # # Local reconstructor
        # {
        #     "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
        #     "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
        #     "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
        #     "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_only_local_0.5'",
        #     "log_dir": os.path.join("logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_only_local_0.5"),
        #     "video_only": True,
        # },
        # Global reconstructor
        # {
        #     "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
        #     "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
        #     "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
        #     "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_only_global_0.5'",
        #     "log_dir": os.path.join("logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_only_global_0.5"),
        #     "video_only": True,
        # },
    ]

    av_v_global_local_exp = [
        # NO reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "none"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_none_{args.lr}",
            "log_dir": os.path.join("logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_none_{args.lr}"),
        },
        # Local reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_local_{args.lr}",
            "log_dir": os.path.join("logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_local_{args.lr}"),
        },
        # Global reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_global_{args.lr}",
            "log_dir": os.path.join(
                "logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_global_{args.lr}"
            ),
        },
    ]

    av_av_global_local_exp = [
        # NO reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "none"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_audio_none_{args.lr}",
            "log_dir": os.path.join(
                "logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_audio_none_{args.lr}"
            ),
        },
        # Local reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_audio_local_{args.lr}",
            "log_dir": os.path.join(
                "logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_audio_local_{args.lr}"
            ),
        },
        # Global reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.5},
            "checkpoint_name": f"SA-LSTM_{args.epochs}_epochs_video_audio_global_{args.lr}",
            "log_dir": os.path.join(
                "logs", dataset, f"SA-LSTM_{args.epochs}_epochs_video_audio_global_{args.lr}"
            ),
        },
    ]

    experiments = [*v_v_global_local_exp, av_v_global_local_exp, av_av_global_local_exp]

    print()
    print(f"Performing {len(experiments)} experiments")
    print()

    for exp in experiments:
        train_config = TrainerConfig()

        # Training params
        train_config.batch_size = exp["training"]["batch_size"]
        train_config.epochs = exp["training"]["epochs"]
        train_config.lr = exp["training"]["lr"]

        # Loss terms
        train_config.reg_lambda = exp["loss"]["reg_lambda"]
        train_config.audio_recon_lambda = exp["loss"]["audio_recon_lambda"]
        train_config.visual_recon_lambda = exp["loss"]["visual_recon_lambda"]

        train_loader, train_dataset = get_loader(
            root_dir=dataset_folder,
            dataset=dataset,
            split="train",
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            video_only=exp["video_only"] if "video_only" in exp else False,
        )
        val_loader, _ = get_loader(
            root_dir=dataset_folder,
            dataset=dataset,
            split="val",
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            video_only=exp["video_only"] if "video_only" in exp else False,
        )
        test_loader, _ = get_loader(
            root_dir=dataset_folder,
            dataset=dataset,
            split="val",  # "test" FIXME: MSR-VTT test fails
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            video_only=exp["video_only"] if "video_only" in exp else False,
        )

        if dual:
            model = AVCaptioningDual(
                vocab=vocab,
                teacher_forcing_ratio=exp["model"]["teacher_forcing_ratio"],
                reconstructor_type=exp["model"]["reconstructor_type"],
                device=device,
            )
        else:
            model = AVCaptioning(
                vocab=vocab,
                teacher_forcing_ratio=exp["model"]["teacher_forcing_ratio"],
                reconstructor_type=exp["model"]["reconstructor_type"],
                device=device,
            )

        # Try loading previous checkpoint (transfer learning)
        base_checkpoint_name = (
            os.path.join(CHECKPOINTS_DIR, exp["base_checkpoint_name"]) if "base_checkpoint_name" in exp else None
        )
        if base_checkpoint_name and os.path.isdir(base_checkpoint_name):
            checkpoint = torch.load()

        model.to(device)

        print("Start training")
        print(json.dumps(exp, sort_keys=True, indent=4))

        if not os.path.isdir(exp["log_dir"]):
            os.makedirs(exp["log_dir"])

        checkpoint_name = os.path.join(CHECKPOINTS_DIR, exp["checkpoint_name"] + ".ckpt")
        tr = Trainer(checkpoint_name=checkpoint_name, log_dir=exp["log_dir"], eval_freq=1)
        history = tr.fit(model, train_loader, val_loader, test_loader, device, train_config,)

        # Save the history where the checkpoint is
        with open(checkpoint_name.replace(".ckpt", ".json"), "w") as f:
            json.dump(history, f)
