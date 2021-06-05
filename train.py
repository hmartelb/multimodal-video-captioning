import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from get_loader import get_loader
from model import CNNtoRNN
from utils import load_checkpoint, print_examples, save_checkpoint


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 1e-4
    num_epochs = 30

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0
    display_freq = 10

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    best_loss = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}:")
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)
        total_loss = 0

        progress = tqdm(
            enumerate(train_loader), total=len(train_loader)
        )
        for idx, (imgs, captions) in progress:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            total_loss += loss.item()

            if idx % display_freq == 0:
                progress.set_postfix({
                    'loss (CE)': float(total_loss/(idx+1)),
                })

        
        if save_model and (best_loss is None or total_loss < best_loss):
            print(f"Loss improved from {best_loss} to {total_loss}.")
            best_loss = total_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)
            print_examples(model, device, dataset)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    # Select which GPU to use and enable mixed precision
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train()
