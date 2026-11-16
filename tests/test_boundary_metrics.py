import pytest
import torch

from cropland.metrics import BoundaryMetrics


def rectangle(left):
    mask = torch.zeros(32, 40)
    mask[8:24, left:left + 12] = 1
    return mask


def test_boundary_matching_tolerance_and_empty_cases():
    reference = rectangle(8)
    identical = BoundaryMetrics()
    identical.update(reference, reference)
    assert identical.compute()["boundary_f1"] == 1
    shifted = rectangle(10)
    tolerant, exact = BoundaryMetrics(2), BoundaryMetrics(0)
    tolerant.update(shifted, reference)
    exact.update(shifted, reference)
    assert tolerant.compute()["boundary_f1"] == 1
    assert exact.compute()["boundary_f1"] < 1
    empty = torch.zeros_like(reference)
    for p, y, expected in ((empty, empty, 1), (empty, reference, 0), (reference, empty, 0)):
        metric = BoundaryMetrics()
        metric.update(p, y)
        assert metric.compute()["boundary_f1"] == expected


def test_boundary_metrics_accumulate_counts_instead_of_image_scores():
    metric = BoundaryMetrics(0)
    first, second = rectangle(8), rectangle(20)
    metric.update(first, first)
    metric.update(torch.zeros_like(second), second)
    result = metric.compute()
    assert result["boundary_precision"] == 1
    assert result["boundary_recall"] == pytest.approx(.5)
    assert result["boundary_f1"] == pytest.approx(2 / 3)
