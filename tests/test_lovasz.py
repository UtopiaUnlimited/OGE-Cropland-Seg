import pytest
import torch

from cropland.common import load_config
from cropland.losses import bce_dice, boundary_target, combined_loss, lovasz_hinge


def test_lovasz_hand_calculated_value_and_gradient():
    logits = torch.tensor([.2, -.3, .7, -1.2], requires_grad=True)
    labels = torch.tensor([1., 0., 0., 1.])
    loss = lovasz_hinge(logits, labels)
    assert loss.item() == pytest.approx(1.65)
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([-1/3, 0., 1/6, -.5]))


@pytest.mark.parametrize("label", [0., 1.])
def test_lovasz_empty_or_full_foreground_has_finite_useful_gradient(label):
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    loss = lovasz_hinge(logits, torch.full_like(logits, label))
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.sum() > 0) if label == 0 else (logits.grad.sum() < 0)
    correct = torch.full_like(logits, 2. if label else -2.)
    assert lovasz_hinge(correct, torch.full_like(logits, label)).item() == 0.


def test_lovasz_pools_batch_and_keeps_background_valid():
    logits = torch.tensor([.2, -.3, .7, -1.2]).reshape(2, 1, 1, 2)
    labels = torch.tensor([1., 0., 0., 1.]).reshape_as(logits)
    assert lovasz_hinge(logits, labels).item() == pytest.approx(1.65)
    assert lovasz_hinge(logits.flip(0), labels.flip(0)).item() == pytest.approx(1.65)
    per_image = torch.stack([lovasz_hinge(x, y) for x, y in zip(logits, labels)]).mean()
    assert not torch.isclose(lovasz_hinge(logits, labels), per_image)


def test_lovasz_autocast_uses_float32_and_backpropagates():
    logits = torch.tensor([.2, -.3, .7, -1.2], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([1., 0., 0., 1.])
    expected = lovasz_hinge(logits.float(), labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = lovasz_hinge(logits, labels)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_combined_loss_applies_lovasz_to_main_and_auxiliary_only():
    main = torch.tensor([.2, -.3, .7, -1.2]).reshape(1, 1, 2, 2).requires_grad_()
    aux = (-main.detach()).requires_grad_()
    boundary = torch.zeros_like(main, requires_grad=True)
    labels = torch.tensor([1., 0., 0., 1.]).reshape_as(main)
    output = dict(logits=main, auxiliary_logits=aux, auxiliary_weight=.4, boundary=boundary)
    original = bce_dice(main, labels) + .4 * bce_dice(aux, labels) + .2 * bce_dice(boundary, boundary_target(labels))
    assert torch.equal(combined_loss(output, labels, .2), original)
    actual = combined_loss(output, labels, .2, 1.)
    expected = original + lovasz_hinge(main, labels) + .4 * lovasz_hinge(aux, labels)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (main, aux, boundary))


def test_e4_lovasz_is_an_isolated_loss_only_experiment():
    from cropland.common import ROOT
    original = load_config(ROOT / "configs/e4_upernet_r50.yaml")
    variant = load_config(ROOT / "configs/e4_upernet_r50_lovasz.yaml")
    assert variant.pop("lovasz_weight") == 1.
    assert variant.pop("run_dir") != original.pop("run_dir")
    assert original == variant


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), "1"])
def test_training_rejects_invalid_lovasz_weight_before_creating_outputs(weight):
    from cropland.train import train
    with pytest.raises(ValueError, match="lovasz_weight"):
        train(dict(lovasz_weight=weight))
