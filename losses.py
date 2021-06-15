import torch
import torch.nn as nn
import torch.nn.functional as F


def EntropyLoss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0  # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean()  # Sum along words and mean along batch
    return b


def GlobalReconstructionLoss(x, x_recon, keep_mask):
    x = x.mean(dim=1)

    caption_len = keep_mask.sum(dim=0)
    caption_len = caption_len.unsqueeze(1).expand(caption_len.size(0), x_recon.size(2))
    caption_len = caption_len.type(torch.cuda.FloatTensor)

    keep_mask = keep_mask.transpose(0, 1).unsqueeze(2).expand_as(x_recon).type(torch.cuda.FloatTensor)

    x_recon = keep_mask * x_recon
    x_recon = x_recon.sum(dim=1) / caption_len

    return F.mse_loss(x, x_recon)


def LocalReconstructionLoss(x, x_recon):
    return F.mse_loss(x, x_recon)


# def ReconstructionLoss(mode='none'):
#     assert mode in ['none', 'global', 'local'], "Wrong mode specified, must be one of ['none', 'global', 'local']"

#     def f(x, x_recon, keep_mask=None):
#         if mode == 'global':
#             return
#         if mode == 'local':
#             return
#         return torch.zeros(1)
#     return f


def TotalReconstructionLoss(
    output,
    captions,
    vocab,
    features=None,
    features_recons=None,
    reg_lambda=0,
    recon_lambda=0,
    # loss_type=None,
    reconstruction_type='global',
):
    PAD_idx = vocab.stoi["<PAD>"]

    # Cross entropy loss
    cross_entropy_loss = F.nll_loss(
        output[1:].view(-1, len(vocab)), captions[1:].contiguous().view(-1), ignore_index=PAD_idx
    )
    # Entropy loss
    entropy_loss = EntropyLoss(output[1:], ignore_mask=(captions[1:] == PAD_idx))

    # Reconstruction loss
    if features_recons is None:
        reconstruction_loss = torch.zeros(1)
    else:
        if reconstruction_type == "global":
            reconstruction_loss = GlobalReconstructionLoss(features, features_recons, keep_mask=(captions != PAD_idx))
        else:
            reconstruction_loss = LocalReconstructionLoss(features, features_recons)

    # Total loss
    loss = cross_entropy_loss + (reg_lambda * entropy_loss) + (recon_lambda * reconstruction_loss)
    return loss, cross_entropy_loss, entropy_loss, reconstruction_loss
