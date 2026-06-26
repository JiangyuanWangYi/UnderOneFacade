#!/usr/bin/env python
"""
Train a model on source country data.
Usage:
    python train.py --model pointnet2 --lofg lofg2 --features xyz \
        --train_countries nottingham --config config.yaml
"""

import os
import sys
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.misc import (
    load_config, set_seed, get_device, get_label_info,
    get_feature_info, get_data_paths, discover_files, resolve_weight_path,
)
from utils.metrics import compute_metrics, print_results, save_results
from models import get_engine


def parse_args():
    p = argparse.ArgumentParser("Cross-continent facade training")
    p.add_argument("--model", required=True, help="Model name (pointnet2, dgcnn, ptv1, ptv3, octformer)")
    p.add_argument("--lofg", required=True, choices=["lofg2", "lofg3"])
    p.add_argument("--features", default="xyz", choices=["xyz", "rgbi"])
    p.add_argument("--train_countries", required=True, help="Comma-separated countries for training")
    p.add_argument("--val_countries", default=None, help="Comma-separated countries for validation (default: same as train)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--out_dir", default=None, help="Output directory (for resuming in original folder)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    engine = get_engine(args.model)

    # Setup
    set_seed(cfg["training"]["seed"], cfg["training"].get("deterministic", False))
    device = get_device(args.device)
    label_info = get_label_info(cfg, args.lofg)
    feat_info = get_feature_info(cfg, args.features)
    num_classes = label_info["num_classes"]
    in_channels = feat_info["num_channels"]
    feature_columns = feat_info["columns"]
    label_offset = cfg.get("label_offset", 1)
    mcfg = cfg["models"][args.model]

    # Output directory
    source_key = "_".join(sorted(args.train_countries.split(",")))
    run_name = f"{args.model}_{args.lofg}_{args.features}_{source_key}"
    if args.out_dir:
        out_dir = args.out_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(cfg["output"]["root"], "train", f"{run_name}_{timestamp}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Determine validation countries
    val_countries = args.val_countries if args.val_countries else args.train_countries
    
    print(f"Model: {args.model} | LoFG: {args.lofg} | Features: {args.features}")
    print(f"Train countries: {args.train_countries} | Device: {device}")
    print(f"Val countries: {val_countries}")
    print(f"Classes: {num_classes} | Channels: {in_channels}")
    print(f"Output: {out_dir}")

    # Data
    train_dirs = get_data_paths(cfg, args.train_countries, "train")
    val_dirs = get_data_paths(cfg, val_countries, "val")
    train_files = discover_files(train_dirs)
    val_files = discover_files(val_dirs)
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")
    if len(train_files) == 0:
        raise SystemExit(
            "No training files found. Check config data.root and country paths. "
            f"Train dirs used: {[d['path'] for d in train_dirs]}"
        )
    if len(val_files) == 0:
        raise SystemExit(
            "No validation files found. Check config data.root and country paths. "
            f"Val dirs used: {[d['path'] for d in val_dirs]}"
        )

    train_ds = engine.build_dataset(train_files, cfg, "train", args.lofg, feature_columns, label_offset)
    val_ds = engine.build_dataset(val_files, cfg, "val", args.lofg, feature_columns, label_offset)

    nw = cfg["training"].get("num_workers", 4)
    bs = engine.get_batch_size(cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              drop_last=(len(train_ds) > bs),
                              collate_fn=engine.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True, drop_last=False,
                            collate_fn=engine.collate_fn)

    # Model
    model = engine.build_model(
        num_classes, in_channels, model_cfg=mcfg
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Optimizer, scheduler, criterion
    optimizer = engine.get_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = engine.get_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
    step_per_batch = mcfg.get("scheduler", "multistep") == "onecycle"
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    epochs = engine.get_epochs(cfg)

    # AMP (Automatic Mixed Precision)
    enable_amp = mcfg.get("enable_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    amp_dtype = torch.float16 if mcfg.get("amp_dtype", "float16") == "float16" else torch.bfloat16

    # Gradient accumulation: accumulate over N mini-batches to emulate a larger effective batch
    accum_steps = mcfg.get("grad_accum_steps", 1)
    grad_clip = mcfg.get("grad_clip", 0.0)
    print(f"Gradient accumulation steps: {accum_steps} (effective batch: {bs * accum_steps})")

    # Resume
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_oa = 0.0
    best_val_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except (RuntimeError, Exception) as e:
            print(f"Corrupted checkpoint (not a valid .pth): {args.resume}")
            print(f"  {e}")
            sys.exit(1)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    class_names = label_info.get("names", [f"class_{i}" for i in range(num_classes)])

    for epoch in range(start_epoch, epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n**** Epoch {epoch+1} ({epoch+1}/{epochs}) ****")
        print(f"Learning rate: {lr_now:.6f}")

        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}/{epochs}")):
            is_last_accum = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            try:
                with torch.cuda.amp.autocast(enabled=enable_amp, dtype=amp_dtype):
                    loss, pred_flat, lbl_flat = engine.train_step(model, batch, criterion, device)
                # Scale loss by accum_steps so gradients sum to the correct magnitude
                scaler.scale(loss / accum_steps).backward()

                if is_last_accum:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    torch.cuda.synchronize()  # catch CUDA errors early
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if step_per_batch:
                        scheduler.step()

                total_loss += loss.item()
                all_preds.append(pred_flat.argmax(1).detach().cpu().numpy())
                all_labels.append(lbl_flat.detach().cpu().numpy())
            except RuntimeError as e:
                print(f"\n[ERROR] Epoch {epoch+1}, Batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                raise

        if not step_per_batch:
            scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        if not all_preds:
            print(f"[WARN] Epoch {epoch+1}: no training batches produced (dataset size={len(train_ds)}, batch_size={bs}). Skipping.")
            continue
        p_cat = np.concatenate(all_preds)
        l_cat = np.concatenate(all_labels)
        train_m = compute_metrics(p_cat, l_cat, num_classes)

        print(f"Training mean loss: {avg_loss:.6f}")
        print(f"Training accuracy: {train_m['OA']:.6f}")
        print(f"Training avg class IoU: {train_m['mIoU']:.6f}")
        print(f"Training avg class F1: {train_m['mF1']:.6f}")

        # Validation
        print(f"---- EPOCH {epoch+1:03d} EVALUATION ----")
        model.eval()
        val_preds, val_labels = [], []
        for batch in tqdm(val_loader, desc=f"Val   E{epoch+1}/{epochs}"):
            pred_flat, lbl_flat = engine.eval_step(model, batch, device)
            val_preds.append(pred_flat.argmax(1).cpu().numpy())
            val_labels.append(lbl_flat.cpu().numpy())

        vp = np.concatenate(val_preds)
        vl = np.concatenate(val_labels)
        val_m = compute_metrics(vp, vl, num_classes)

        print(f"eval point accuracy: {val_m['OA']:.6f}")
        print(f"eval point avg class IoU: {val_m['mIoU']:.6f}")
        print(f"eval point avg class F1: {val_m['mF1']:.6f}")
        print(f"eval point avg class precision: {val_m['mP']:.6f}")
        print(f"eval point avg class recall: {val_m['mR']:.6f}")

        # Checkpointing
        save_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
        }
        os.makedirs(ckpt_dir, exist_ok=True)  # Ensure dir exists before save
        torch.save(save_state, os.path.join(ckpt_dir, "latest.pth"))

        if (epoch + 1) % 10 == 0:
            torch.save(save_state, os.path.join(ckpt_dir, f"ckpt_e{epoch+1:04d}.pth"))

        if val_m["mF1"] > best_val_f1:
            best_val_f1 = val_m["mF1"]
            best_val_oa = val_m["OA"]
            best_val_miou = val_m["mIoU"]
            torch.save(save_state, os.path.join(ckpt_dir, "best.pth"))
            print(f"  *** New best val mF1: {100*best_val_f1:.2f}% ***")

    # Final val evaluation with best checkpoint
    best_ckpt = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    val_preds, val_labels = [], []
    for batch in tqdm(val_loader, desc="Final Val (best ckpt)", leave=False):
        pred_flat, lbl_flat = engine.eval_step(model, batch, device)
        val_preds.append(pred_flat.argmax(1).cpu().numpy())
        val_labels.append(lbl_flat.cpu().numpy())

    final_m = compute_metrics(np.concatenate(val_preds), np.concatenate(val_labels), num_classes)
    class_names = label_info.get("names", [f"class_{i}" for i in range(num_classes)])
    print_results(final_m, class_names, prefix=f"Train {source_key} (best ckpt)")
    save_results(final_m, class_names,
                 os.path.join(out_dir, "val_results.txt"),
                 header=f"Train: {args.train_countries} | Val: {val_countries}")

    # Save best weights to standardized location
    if os.path.exists(best_ckpt):
        dst = resolve_weight_path(
            cfg["weights"]["root"], args.train_countries, args.model, args.lofg, args.features
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        import shutil
        shutil.copy2(best_ckpt, dst)
        print(f"Best weights saved to: {dst}")

    print(f"Training complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
