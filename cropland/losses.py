import torch
import torch.nn.functional as F


def bce_dice(logits, target):
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1 - ((2 * (p * target).sum(dims) + 1) / (p.sum(dims) + target.sum(dims) + 1)).mean()
    return bce + dice


def boundary_target(mask):
    # Replicate padding avoids inventing a border around an all-foreground image.
    padded = F.pad(mask.float(), (1, 1, 1, 1), mode="replicate")
    return F.max_pool2d(padded, 3, 1) + F.max_pool2d(-padded, 3, 1)


def combined_loss(output, target, boundary_weight=0):
    semantic = bce_dice(output["logits"], target)
    if boundary_weight:
        return semantic + boundary_weight * bce_dice(output["boundary"], boundary_target(target))
    return semantic
