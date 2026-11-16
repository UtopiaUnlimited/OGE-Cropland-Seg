import torch.nn.functional as F

from .losses import boundary_target


class BinaryMetrics:
    """Dataset-wide confusion counts (never average per-batch IoU)."""
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, prediction, target):
        p, y = prediction.detach().bool(), target.detach().bool()
        self.tp += int((p & y).sum())
        self.fp += int((p & ~y).sum())
        self.fn += int((~p & y).sum())
        self.tn += int((~p & ~y).sum())

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        ratio = lambda n, d: n / d if d else 1.0
        iou = ratio(tp, tp + fp + fn)
        bg = ratio(tn, tn + fp + fn)
        return dict(iou=iou, background_iou=bg, miou=(iou + bg) / 2,
                    f1=ratio(2 * tp, 2 * tp + fp + fn), precision=ratio(tp, tp + fp),
                    recall=ratio(tp, tp + fn), accuracy=ratio(tp + tn, tp + tn + fp + fn),
                    tp=tp, fp=fp, fn=fn, tn=tn)


class BoundaryMetrics:
    """Match mask boundary pixels within a square pixel neighborhood."""

    def __init__(self, tolerance=2):
        self.tolerance = tolerance
        self.predicted = self.actual = 0
        self.matched_predicted = self.matched_actual = 0

    def update(self, prediction, target):
        p = boundary_target(prediction.detach()[None, None])
        y = boundary_target(target.detach()[None, None])
        radius = self.tolerance
        near_y = F.max_pool2d(y, 2 * radius + 1, stride=1, padding=radius)
        near_p = F.max_pool2d(p, 2 * radius + 1, stride=1, padding=radius)
        self.predicted += int(p.sum())
        self.actual += int(y.sum())
        self.matched_predicted += int((p * near_y).sum())
        self.matched_actual += int((y * near_p).sum())

    def compute(self):
        precision = self.matched_predicted / self.predicted if self.predicted else 1.0
        recall = self.matched_actual / self.actual if self.actual else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return dict(boundary_precision=precision, boundary_recall=recall, boundary_f1=f1)
