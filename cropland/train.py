from __future__ import annotations

import argparse
import hashlib
import math
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from .common import get_device, load_config, load_json, save_json, seed_everything, seed_worker
from .data import FarmDataset, read_rows
from .losses import combined_loss
from .metrics import BinaryMetrics
from .model import build_model, load_checkpoint, save_checkpoint


def split_fingerprints(directory):
    result, names, hashes = {}, set(), set()
    for name in ("train", "val", "test"):
        path = Path(directory) / f"{name}.csv"
        rows = read_rows(path)
        current = {r["filename"] for r in rows}
        current_hashes = {r["image_sha256"] for r in rows}
        if current & names or current_hashes & hashes:
            raise ValueError(f"Data leakage: overlapping {name} split")
        names |= current
        hashes |= current_hashes
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def accumulation_weight(batch_index, batch_count, batch_size, dataset_size, accumulation_steps, actual_batch_size):
    group_start = batch_index // accumulation_steps * accumulation_steps
    group_batches = min(accumulation_steps, batch_count - group_start)
    group_samples = min(group_batches * batch_size, dataset_size - group_start * batch_size)
    return actual_batch_size / group_samples


def validate(model, loader, device, amp, boundary_weight, max_batches=None):
    model.eval()
    metrics, total, samples = BinaryMetrics(), 0., 0
    preview = None
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = batch["image"].to(device), batch["mask"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(x)
                loss = combined_loss(output, y, boundary_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            p = output["logits"].float().sigmoid()
            metrics.update(p >= .5, y)
            total += loss.item() * len(x)
            samples += len(x)
            if preview is None:
                preview = (x[0].cpu(), y[0].cpu(), p[0].cpu())
    return {**metrics.compute(), "loss": total / samples}, preview


def train(cfg, resume=None):
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    # Large CPU thread counts slow down small test batches substantially.
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    amp = cfg["amp"] and device.type == "cuda"
    prep = load_json(cfg["preprocessing"])
    fingerprints = split_fingerprints(cfg["split_dir"])
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    if (run / "last.pt").exists() and resume is None:
        raise FileExistsError(f"Run already has checkpoints: {run}; use --resume or a different run_dir")
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {}
    for split in ("train", "val"):
        dataset = FarmDataset(cfg["data_root"], Path(cfg["split_dir"]) / f"{split}.csv",
                              prep, cfg["image_size"], augment=split == "train")
        loaders[split] = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=split == "train",
                                     num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                                     worker_init_fn=seed_worker, generator=generator if split == "train" else None)
    checkpoint = None
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires last.pt with optimizer state; best.pt is inference-only")
        if checkpoint["split_fingerprints"] != fingerprints or checkpoint["preprocessing"] != prep:
            raise ValueError("Resume requires identical splits and preprocessing")
        for key in ("image_size", "batch_size", "accumulation_steps", "epochs", "learning_rate", "weight_decay",
                    "warmup_epochs", "boundary_weight", "freeze_encoder", "seed", "max_train_batches", "max_val_batches"):
            if cfg[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume configuration changed: {key}")
        if cfg["freeze_encoder"]:
            model.semantic.segformer.requires_grad_(False)
    else:
        model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    batch_count = min(len(loaders["train"]), cfg.get("max_train_batches") or len(loaders["train"]))
    steps_per_epoch = math.ceil(batch_count / cfg["accumulation_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    # A conservative initial scale avoids skipping the first steps of the deep B5 encoder.
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.)
    start, best, stale, global_step = 0, -1., 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, stale, global_step = checkpoint["epoch"] + 1, checkpoint["best_iou"], checkpoint["stale"], checkpoint["global_step"]
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    save_json(run / "resolved_config.json", cfg)
    save_json(run / "split_fingerprints.json", fingerprints)
    print(f"Device={device}, AMP={amp}, model=MiT-B5, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    history_path = run / "history.json"
    history = load_json(history_path) if resume and history_path.exists() else []
    with SummaryWriter(str(run / "tensorboard"), purge_step=start if resume else None) as writer:
        for epoch in range(start, cfg["epochs"]):
            epoch_step_start = global_step
            model.train()
            if cfg["freeze_encoder"]:
                model.semantic.segformer.eval()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            total, seen = 0., 0
            bar = tqdm(loaders["train"], total=batch_count, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
            for i, batch in enumerate(bar):
                if i >= batch_count:
                    break
                x, y = batch["image"].to(device), batch["mask"].to(device)
                # Weight by actual samples, including incomplete final batches/groups.
                weight = accumulation_weight(i, batch_count, cfg["batch_size"], len(loaders["train"].dataset),
                                             cfg["accumulation_steps"], len(x))
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = model(x)
                    loss = combined_loss(output, y, cfg["boundary_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss * weight).backward()
                if (i + 1) % cfg["accumulation_steps"] == 0 or i + 1 == batch_count:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item() * len(x)
                seen += len(x)
                bar.set_postfix(loss=f"{loss.item():.4f}")
            stats, preview = validate(model, loaders["val"], device, amp, cfg["boundary_weight"], cfg.get("max_val_batches"))
            improved = stats["iou"] > best
            best, stale = max(best, stats["iou"]), 0 if improved else stale + 1
            metrics = dict(epoch=epoch + 1, train_loss=total / seen, val=stats,
                           optimizer_steps=global_step - epoch_step_start,
                           amp_scale=scaler.get_scale(),
                           learning_rate=optimizer.param_groups[0]["lr"],
                           peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.)
            history.append(metrics)
            save_json(history_path, history)
            writer.add_scalar("loss/train", total / seen, epoch)
            for key, value in stats.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("optimizer/steps", metrics["optimizer_steps"], epoch)
            writer.add_scalar("optimizer/amp_scale", metrics["amp_scale"], epoch)
            writer.add_scalar("gpu/peak_memory_mb", metrics["peak_gpu_memory_mb"], epoch)
            image, truth, probability = preview
            image = (image * torch.tensor(prep["std"])[:, None, None] + torch.tensor(prep["mean"])[:, None, None]).clamp(0, 1)
            writer.add_image("val/image", image, epoch)
            writer.add_image("val/ground_truth", truth, epoch)
            writer.add_image("val/probability", probability, epoch)
            writer.flush()
            common = dict(epoch=epoch, best_iou=best, stale=stale, global_step=global_step,
                          split_fingerprints=fingerprints)
            if improved:
                save_checkpoint(run / "best.pt", model, prep, cfg, **common)
            save_checkpoint(run / "last.pt", model, prep, cfg, **common,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            scaler=scaler.state_dict(), loader_rng=generator.get_state(),
                            torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            print(metrics, flush=True)
            if stale >= cfg["patience"]:
                print("Early stopping", flush=True)
                break
    if global_step == 0:
        raise RuntimeError("No optimizer updates occurred. Check AMP overflow or disable amp; this run is not trained.")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer-B5 with BCE + Dice and optional boundary supervision")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="Resume from last.pt (best.pt is inference-only)")
    args = parser.parse_args()
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
