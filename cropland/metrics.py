import torch


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
