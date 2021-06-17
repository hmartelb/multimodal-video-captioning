import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def EntropyLoss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0  # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean()  # Sum along words and mean along batch
    return b


def GlobalReconstructionLoss(x, x_recon, keep_mask):
    # Quick fix, make sure everyting is in the same device (CPU)
    # x = x.to(x_recon.device)
    x_recon = x_recon.to(x.device)
    keep_mask = keep_mask.to(x.device)

    x = x.mean(dim=1)
    # print(x.shape, x_recon.shape, keep_mask.shape)

    caption_len = keep_mask.sum(dim=0)
    caption_len = caption_len.unsqueeze(1).expand(caption_len.size(0), x_recon.size(2))
    keep_mask = keep_mask.transpose(0, 1).unsqueeze(2).expand_as(x_recon)

    caption_len = caption_len.type(torch.FloatTensor).to(x_recon.device)
    keep_mask = keep_mask.type(torch.FloatTensor).to(x_recon.device)

    x_recon = keep_mask * x_recon
    x_recon = x_recon.sum(dim=1) / caption_len

    # print(x.device, x_recon.device, keep_mask.device)

    return F.mse_loss(x, x_recon)


def LocalReconstructionLoss(x, x_recon):
    return F.mse_loss(x, x_recon)


def TotalReconstructionLoss(
    output,
    captions,
    features=None,
    features_recons=None,
    reg_lambda=0,
    recon_lambda=0,
    reconstruction_type="global",
):
    PAD_idx = 0  # vocab.stoi["<PAD>"] = 0

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
    if features_recons is None:
        reconstruction_loss = torch.zeros(1).to(output.device)
    else:
        if reconstruction_type == "global":
            reconstruction_loss = GlobalReconstructionLoss(features, features_recons, keep_mask=(captions != PAD_idx))
        elif reconstruction_type == "local":
            reconstruction_loss = LocalReconstructionLoss(features, features_recons)
        else:
            reconstruction_loss = torch.zeros(1).to(output.device)

    # print(type(cross_entropy_loss), type(reg_lambda), type(entropy_loss), type(recon_lambda), type(reconstruction_loss))

    # Total loss
    loss = cross_entropy_loss + (reg_lambda * entropy_loss) + (recon_lambda * reconstruction_loss)
    return loss, cross_entropy_loss, entropy_loss, reconstruction_loss


def ReconstructionLossBuilder(reg_lambda, recon_lambda, reconstruction_type):
    assert reconstruction_type in [
        "none",
        "global",
        "local",
    ], "Wrong mode specified, must be one of ['none', 'global', 'local']"
    return partial(
        TotalReconstructionLoss,
        reg_lambda=reg_lambda,
        recon_lambda=recon_lambda,
        reconstruction_type=reconstruction_type,
    )


def ModalityWiseReconstructionLoss(
    output,
    captions,
    audio_features=None,
    audio_features_recons=None,
    visual_features=None,
    visual_features_recons=None,
    reg_lambda=0,
    audio_recon_lambda=0,
    visual_recon_lambda=0,
    rec_type="none",
):
    def _reconstruction_loss(captions, features, features_recons=None, rec_type="none"):
        # Reconstruction loss
        if features_recons is None:
            rec_loss = torch.zeros(1).to(output.device)
        else:
            if rec_type == "global":
                rec_loss = GlobalReconstructionLoss(features, features_recons, keep_mask=(captions != 0))
            elif rec_type == "local":
                rec_loss = LocalReconstructionLoss(features, features_recons)
            else:
                rec_loss = torch.zeros(1).to(output.device)
        return rec_loss

    vocab_size = int(output.shape[2])

    # Cross entropy loss
    cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab_size), captions[1:].contiguous().view(-1), ignore_index=0)
    # Entropy loss
    entropy_loss = EntropyLoss(output[1:], ignore_mask=(captions[1:] == 0))

    audio_reconstruction_loss = _reconstruction_loss(captions, audio_features, audio_features_recons, rec_type)
    visual_reconstruction_loss = _reconstruction_loss(captions, visual_features, visual_features_recons, rec_type)

    # Total loss with modality-wise reconstructions
    loss = cross_entropy_loss + (reg_lambda * entropy_loss)
    loss += audio_recon_lambda * audio_reconstruction_loss
    loss += visual_recon_lambda * visual_reconstruction_loss

    return loss, cross_entropy_loss, entropy_loss, audio_reconstruction_loss, visual_reconstruction_loss


def ModalityWiseReconstructionLossBuilder(reg_lambda, audio_recon_lambda, visual_recon_lambda, rec_type='none'):
    assert rec_type in [
        "none",
        "global",
        "local",
    ], "Wrong mode specified, must be one of ['none', 'global', 'local']"
    return partial(
        ModalityWiseReconstructionLoss,
        reg_lambda=reg_lambda,
        audio_recon_lambda=audio_recon_lambda,
        visual_recon_lambda=visual_recon_lambda,
        rec_type=rec_type,
    )


def NLPScore(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence) , prediction
    hypo, dictionary of hypothesis sentences (id, sentence), GT
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == "__main__":
    batch_size = 2
    vocab_size = 3056

    audio_feature_size = 128
    visual_feature_size = 1000
    feature_size = audio_feature_size + visual_feature_size

    max_caption_len = 18 + 2
    seconds = 4

    output = torch.rand([max_caption_len, batch_size, vocab_size])
    captions = torch.rand([max_caption_len, batch_size])
    # features = torch.rand([batch_size, seconds, feature_size])
    audio_features, audio_features_recons = torch.rand([batch_size, seconds, audio_feature_size]), torch.rand([batch_size, seconds, audio_feature_size])
    visual_features, visual_features_recons = torch.rand([batch_size, seconds, audio_feature_size]), torch.rand([batch_size, seconds, audio_feature_size])

    loss, cross_entropy_loss, entropy_loss, audio_reconstruction_loss, visual_reconstruction_loss = ModalityWiseReconstructionLoss(
        output,
        captions,
        audio_features,
        audio_features_recons,
        visual_features,
        visual_features_recons,
        reg_lambda=1,
        audio_recon_lambda=1,
        visual_recon_lambda=1,
        rec_type="global",
    )
    print(loss, cross_entropy_loss, entropy_loss, audio_reconstruction_loss, visual_reconstruction_loss)
    print(loss.shape, cross_entropy_loss.shape, entropy_loss.shape, audio_reconstruction_loss.shape, visual_reconstruction_loss.shape)
