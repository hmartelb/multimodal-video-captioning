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
    caption_len = caption_len.type(torch.FloatTensor)

    keep_mask = keep_mask.transpose(0, 1).unsqueeze(2).expand_as(x_recon).type(torch.FloatTensor)

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
    features=None,
    features_recons=None,
    reg_lambda=0,
    recon_lambda=0,
    reconstruction_type='global',
):
    PAD_idx = 0 #vocab.stoi["<PAD>"] = 0

    vocab_size = int(output.shape[2])
    # print(vocab_size)

    # print(output[1:].view(-1, vocab_size).shape)
    # print(captions[1:].contiguous().view(-1).shape)

    # Cross entropy loss
    cross_entropy_loss = F.nll_loss(
        output[1:].view(-1, vocab_size), captions[1:].contiguous().view(-1), ignore_index=PAD_idx
    )
    # Entropy loss
    entropy_loss = EntropyLoss(output[1:], ignore_mask=(captions[1:] == PAD_idx))

    # Reconstruction loss
    # if features_recons is None:
    reconstruction_loss = torch.zeros(1).to('cpu')
    # else:
    #     if reconstruction_type == "global":
    #         reconstruction_loss = GlobalReconstructionLoss(features, features_recons, keep_mask=(captions != PAD_idx))
    #     else:
    #         reconstruction_loss = LocalReconstructionLoss(features, features_recons)

    # print(type(cross_entropy_loss), type(reg_lambda), type(entropy_loss), type(recon_lambda), type(reconstruction_loss))

    # Total loss
    loss = cross_entropy_loss + (reg_lambda * entropy_loss) + (recon_lambda * reconstruction_loss)
    return loss, cross_entropy_loss, entropy_loss, reconstruction_loss

if __name__ == '__main__':
    batch_size = 2
    vocab_size = 3056
    feature_size = 1128
    max_caption_len = 18 + 2
    seconds = 28

    output = torch.rand([max_caption_len, batch_size, vocab_size])
    captions = torch.rand([max_caption_len, batch_size])
    features = torch.rand([batch_size, seconds, feature_size])

    loss, cross_entropy_loss, entropy_loss, reconstruction_loss = TotalReconstructionLoss(
        output, 
        captions,
        features,
    )
    print(loss.shape, cross_entropy_loss.shape, entropy_loss.shape, reconstruction_loss.shape)