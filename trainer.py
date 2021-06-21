import argparse
import gc
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from get_loader import (VideoDataset_to_VideoCaptionsLoader, Vocabulary,
                        get_loader)
from losses import (ModalityWiseReconstructionLoss,
                    ModalityWiseReconstructionLossBuilder, NLPScore,
                    ReconstructionLossBuilder, TotalReconstructionLoss)
from models import AVCaptioning

DEBUG = False  ## FIXME: Set to False before training

class TrainerConfig:
    batch_size = 128

    epochs = 2 if DEBUG else 50
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
    def __init__(self, checkpoint_name, log_dir='logs', display_freq=10, eval_freq=10):
        # assert checkpoint_name.endswith(".tar"), "The checkpoint file must have .tar extension"
        self.checkpoint_name = checkpoint_name
        self.display_freq = display_freq
        self.eval_freq = eval_freq

        self.summary_writer = SummaryWriter(log_dir)

    def _load_checkpoint(self, model):
        if os.path.isfile(self.checkpoint_name):
            try:
                print(f"Resuming training from checkpoint: {self.checkpoint_name}")
                checkpoint = torch.load(self.checkpoint_name)

                model.decoder.load_state_dict(checkpoint["decoder"])
                if model.reconstructor and checkpoint["reconstructor"]:
                    model.reconstructor.load_state_dict(checkpoint["reconstructor"])

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

        self.history = {"train_loss": [], "train_score": [], "val_loss": [], "val_score": [], "test_loss": [], "test_score": []}
        self.previous_epochs = 0
        self.best_loss = 1e6
        self.best_CIDEr = 0

        model = self._load_checkpoint(model)
        model.to(self.device)

        ## VideoCaptionsDataloader for Evaluation
        train_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(train_loader.dataset, train_config.batch_size)
        val_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(val_loader.dataset, train_config.batch_size)
        test_vidCap_loader = VideoDataset_to_VideoCaptionsLoader(test_loader.dataset, train_config.batch_size)

        # Start training
        for epoch in range(self.previous_epochs + 1, train_config.epochs + 1):
            print(f"\nEpoch {epoch}/{train_config.epochs}:")

            train_loss = self.train(model, train_loader, epoch)
            val_loss = self.test(model, val_loader, 'val', epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if epoch % self.eval_freq == 0 or epoch == train_config.epochs:
                train_score, _, _ = self.eval(model, train_vidCap_loader, 'train', epoch)
                val_score, _, _ = self.eval(model, val_vidCap_loader, 'val', epoch)
                self.history["train_score"].append(train_score)
                self.history["val_score"].append(val_score)

                # Scheduler step, check plateau
                self.lr_scheduler.step(val_score['CIDEr'])

                # Save checkpoint based on EVAL METRIC
                if val_score['CIDEr'] > self.best_CIDEr:
                    print(f"CIDEr improved from {self.best_CIDEr} to {val_score['CIDEr']}.")
                    print(f"Saving checkpoint to: {self.checkpoint_name}")
                    
                    self.best_CIDEr = val_score['CIDEr'] 
                    self._save_checkpoint(epoch, model, {})
                    torch.save(model, self.checkpoint_name.replace('.ckpt', '_best.pt'))

            # Save checkpoint only if the validation loss improves (avoid overfitting)
            if val_loss["total"] < self.best_loss:
                print(f"Validation loss improved from {self.best_loss} to {val_loss['total']}.")
                print(f"Saving checkpoint to: {self.checkpoint_name}")

                self.best_loss = val_loss["total"]
                self._save_checkpoint(epoch, model, {})  # FIXME: empty config
                torch.save(model, self.checkpoint_name.replace('.ckpt', '_best.pt'))

        torch.save(model, self.checkpoint_name.replace('.ckpt', '_last.pt'))

        # Evaluate on TEST set using the best model (from checkpoint)
        # model = self._load_checkpoint(model)
        model = torch.load(self.checkpoint_name.replace('.ckpt', '_best.pt'))
        model = model.to(self.device)

        test_loss = self.test(model, test_loader, 'test', epoch)
        test_score, _, _ = self.eval(model, test_vidCap_loader, 'test', epoch, mode='direct')
        # test_score_beam, _, _ = self.eval(model, test_vidCap_loader, 'test', epoch, mode='beam')

        self.history["test_loss"].append(test_loss)
        self.history["test_score"].append(test_score)
        # self.history["test_score"].append(test_score_beam)

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
                    outputs,
                    captions,
                    audio_features,
                    audio_recons,
                    visual_features,
                    visual_recons,
                )
                loss.mean().backward()

                step_no = epoch * len(dataloader) + i
                self.summary_writer.add_scalar('train/loss', loss.mean().item(), step_no)
                self.summary_writer.add_scalar('train/loss/ce', ce.mean().item(), step_no)
                self.summary_writer.add_scalar('train/loss/e', e.mean().item(), step_no)
                self.summary_writer.add_scalar('train/loss/a_recon', a_recon.mean().item(), step_no)
                self.summary_writer.add_scalar('train/loss/v_recon', v_recon.mean().item(), step_no)

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

        self.summary_writer.add_scalar('train_epoch/loss', total_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar('train_epoch/loss/ce', cross_entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar('train_epoch/loss/e',  entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar('train_epoch/loss/a_recon', audio_reconstruction_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar('train_epoch/loss/v_recon', visual_reconstruction_loss / len(dataloader), epoch)

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

                    outputs, audio_recons, visual_recons = model(audio_features, visual_features, captions, teacher_forcing_ratio=0)
                    loss, ce, e, a_recon, v_recon = self.RecLoss(
                        outputs,
                        captions,
                        audio_features,
                        audio_recons,
                        visual_features,
                        visual_recons,
                    )

                    step_no = epoch * len(dataloader) + i
                    self.summary_writer.add_scalar(f'{training_phase}/loss', loss.mean().item(), step_no)
                    self.summary_writer.add_scalar(f'{training_phase}/loss/ce', ce.mean().item(), step_no)
                    self.summary_writer.add_scalar(f'{training_phase}/loss/e', e.mean().item(), step_no)
                    self.summary_writer.add_scalar(f'{training_phase}/loss/a_recon', a_recon.mean().item(), step_no)
                    self.summary_writer.add_scalar(f'{training_phase}/loss/v_recon', v_recon.mean().item(), step_no)

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

        self.summary_writer.add_scalar(f'{training_phase}_epoch/loss', total_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f'{training_phase}_epoch/loss/ce', cross_entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f'{training_phase}_epoch/loss/e',  entropy_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f'{training_phase}_epoch/loss/a_recon', audio_reconstruction_loss / len(dataloader), epoch)
        self.summary_writer.add_scalar(f'{training_phase}_epoch/loss/v_recon', visual_reconstruction_loss / len(dataloader), epoch)

        return {
            "total": total_loss / len(dataloader),
            "ce": cross_entropy_loss / len(dataloader),
            "e": entropy_loss / len(dataloader),
            "a_recon": audio_reconstruction_loss / len(dataloader),
            "v_recon": visual_reconstruction_loss / len(dataloader),
        }

    def eval(self, model, videoCaptions_dataloader, training_phase, epoch, mode='direct', get_scores=True):
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
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/Bleu_1', scores['Bleu_1'], epoch)
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/Bleu_2', scores['Bleu_2'], epoch)
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/Bleu_3', scores['Bleu_3'], epoch)
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/Bleu_4', scores['Bleu_4'], epoch)
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/ROUGE_L', scores['ROUGE_L'], epoch)
            self.summary_writer.add_scalar(f'{training_phase}/score/{mode}/CIDEr', scores['CIDEr'], epoch)
            print(scores)
            
        return scores, vid_GT, vid_gen


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='-1')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_folder = os.path.join("datasets", "MSVD")
    vocab_pkl = os.path.join(dataset_folder, "metadata", "vocab.pkl")
    vocab = Vocabulary.load(vocab_pkl)

    CHECKPOINTS_DIR = os.path.join("checkpoints")

    experiments = [
        # NO reconstructor
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "none"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_forcing_1",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_forcing_1'),
        },
        # {
        #     "model": {"teacher_forcing_ratio": 0.5, "reconstructor_type": "none"},
        #     "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
        #     "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0},
        #     "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_forcing_0.5",
        #     "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_forcing_0.5'),
        # },
        # {
        #     "model": {"teacher_forcing_ratio": 0, "reconstructor_type": "none"},
        #     "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
        #     "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0},
        #     "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_forcing_0",
        #     "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_forcing_0'),
        # },
        # LOCAL reconstructors (video only)
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.1}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_local_0.1",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_local_0.1'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_local_0.5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_local_0.5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 1}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_local_1",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_local_1'),
        },
        # GLOBAL reconstructors (video only)
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4}, 
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.1},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_global_0.1",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_global_0.1'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 0.5},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_global_0.5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_global_0.5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0, "visual_recon_lambda": 1},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_global_1",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_global_1'),
        },
        # LOCAL reconstructors (video + audio)
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.1}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_local_0.1_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_local_0.1_5e-5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.5}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_local_0.5_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_local_0.5_5e-5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "local"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 1}, 
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_local_1_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_local_1_5e-5'),
        },
        # GLOBAL reconstructors (video + audio)
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4}, 
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.1},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_global_0.1_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_global_0.1_5e-5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 0.5},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_global_0.5_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_global_0.5_5e-5'),
        },
        {
            "model": {"teacher_forcing_ratio": 1.0, "reconstructor_type": "global"},
            "training": {"batch_size": 128, "epochs": 50, "lr": 2e-4},
            "loss": {"reg_lambda": 0.0005, "audio_recon_lambda": 0.00005, "visual_recon_lambda": 1},
            "checkpoint_name": "SA-LSTM_50_epochs_reg_5e-4_video_audio_global_1_5e-5",
            "log_dir": os.path.join('logs', 'SA-LSTM_50_epochs_reg_5e-4_video_audio_global_1_5e-5'),
        },
    ]

    #audio_recon_lambda": 0.00005

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
            split="tiny" if DEBUG else "train",
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            # normalize=exp['normalize_dataset'],
        )
        val_loader, _ = get_loader(
            root_dir=dataset_folder,
            split="tiny" if DEBUG else "val",
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            # normalize=exp['normalize_dataset'],
        )
        test_loader, _ = get_loader(
            root_dir=dataset_folder,
            split="tiny" if DEBUG else "test",
            batch_size=train_config.batch_size,
            vocab_pkl=vocab_pkl,
            # normalize=exp['normalize_dataset'],
        )

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
        
        checkpoint_name = os.path.join(CHECKPOINTS_DIR, exp["checkpoint_name"] + ".ckpt")#(".ckpt" if not exp["checkpoint_name"].endswith(".ckpt") else "")
        
        if not os.path.isdir(exp['log_dir']):
            os.makedirs(exp['log_dir'])

        tr = Trainer(checkpoint_name=checkpoint_name, log_dir=exp["log_dir"], eval_freq=1)
        history = tr.fit(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            train_config,
        )
        
        # Save the history where the checkpoint is
        with open(checkpoint_name.replace(".ckpt", ".json"), "w") as f:
            json.dump(history, f)
