import torch
import torch.nn.functional as F


def lovasz_hinge(logits, target):
    """Binary foreground Jaccard surrogate, pooled over the current mini-batch.

    Adapted from Maxim Berman's MIT-licensed LovaszSoftmax implementation
    (2018). See third_party/lovasz/LICENSE and NOTICE.md. Use raw logits,
    not sigmoid probabilities; background (0) remains a valid target.
    """
    if logits.shape != target.shape:
        raise ValueError("Lovasz logits and targets must have the same shape")
    # Sorting/cumulative sums must stay float32 even inside CUDA autocast.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float().reshape(-1)
        labels = target.float().reshape(-1)
        if not labels.numel():
            return scores.sum() * 0.
        errors = 1. - scores * (2. * labels - 1.)
        errors, order = errors.sort(descending=True)
        sorted_labels = labels[order]
        positives = sorted_labels.sum()
        intersection = positives - sorted_labels.cumsum(0)
        union = positives + (1. - sorted_labels).cumsum(0)
        jaccard = 1. - intersection / union
        gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        return torch.dot(F.relu(errors), gradient)


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


def semantic_loss(logits, target, lovasz_weight=0):
    loss = bce_dice(logits, target)
    if lovasz_weight:
        loss = loss + lovasz_weight * lovasz_hinge(logits, target)
    return loss


def combined_loss(output, target, boundary_weight=0, lovasz_weight=0):
    semantic = semantic_loss(output["logits"], target, lovasz_weight)
    if "auxiliary_logits" in output:
        semantic = semantic + output["auxiliary_weight"] * semantic_loss(output["auxiliary_logits"], target, lovasz_weight)
    if boundary_weight:
        return semantic + boundary_weight * bce_dice(output["boundary"], boundary_target(target))
    return semantic
