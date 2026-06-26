import os
import argparse
import datetime
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.misc import (
    load_config, set_seed, get_device, get_label_info,
    get_feature_info, get_data_paths, discover_files, resolve_weight_path,
    infer_in_channels_from_checkpoint, extract_state_dict,
)
from utils.metrics import compute_metrics, print_results, save_results
from models import get_engine


def parse_args():
    p = argparse.ArgumentParser("Cross-continent fine-tuning")
    p.add_argument("--model", required=True)
    p.add_argument("--lofg", required=True, choices=["lofg2", "lofg3"])
    p.add_argument("--features", default="xyz", choices=["xyz", "rgbi"])
    p.add_argument("--pretrained", default="auto", help="Pretrained weights path or 'auto'")
    p.add_argument("--source_countries", required=True, help="Source countries for pretrained weights")
    p.add_argument("--finetune_countries", required=True, help="Target countries for fine-tuning")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--ft_epochs", type=int, default=None, help="Override finetune epochs")
    p.add_argument("--ft_lr_scale", type=float, default=None, help="Override LR scale factor")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    engine = get_engine(args.model)

    set_seed(cfg["training"]["seed"])
    device = get_device(args.device)
    label_info = get_label_info(cfg, args.lofg)
    feat_info = get_feature_info(cfg, args.features)
    num_classes = label_info["num_classes"]
    class_names = label_info["class_names"]
    in_channels = feat_info["num_channels"]
    feature_columns = feat_info["columns"]
    label_offset = cfg.get("label_offset", 1)
    mcfg = cfg["models"][args.model]
    ft_cfg = cfg.get("finetune", {})

    # Resolve pretrained weights
    if args.pretrained == "auto":
        weights_path = resolve_weight_path(
            cfg["weights"]["root"], args.source_countries, args.model, args.lofg, args.features
        )
    else:
        weights_path = args.pretrained

    if not os.path.exists(weights_path):
        print(f"ERROR: pretrained weights not found: {weights_path}")
        return

    source_key = "_".join(sorted(args.source_countries.split(",")))
    target_key = "_".join(sorted(args.finetune_countries.split(",")))
    run_name = f"finetune_{args.model}_{args.lofg}_{args.features}_{source_key}_to_{target_key}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["output"]["root"], "finetune", f"{run_name}_{timestamp}")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    ft_epochs = args.ft_epochs or ft_cfg.get("epochs", 50)
    lr_scale = args.ft_lr_scale or ft_cfg.get("lr_scale", 0.1)
    freeze_epochs = ft_cfg.get("freeze_epochs", 10)

    print(f"Model: {args.model} | LoFG: {args.lofg} | Features: {args.features}")
    print(f"Source: {args.source_countries} -> Fine-tune on: {args.finetune_countries}")
    print(f"Pretrained: {weights_path}")
    print(f"FT epochs: {ft_epochs} | LR scale: {lr_scale} | Freeze epochs: {freeze_epochs}")

    # Data: use target country train/val/test
    train_dirs = get_data_paths(cfg, args.finetune_countries, "train")
    val_dirs = get_data_paths(cfg, args.finetune_countries, "val")
    test_dirs = get_data_paths(cfg, args.finetune_countries, "test")
    train_files = discover_files(train_dirs)
    val_files = discover_files(val_dirs)
    test_files = discover_files(test_dirs)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = engine.build_dataset(train_files, cfg, "train", args.lofg, feature_columns, label_offset)
    val_ds = engine.build_dataset(val_files, cfg, "val", args.lofg, feature_columns, label_offset)
    test_ds = engine.build_dataset(test_files, cfg, "test", args.lofg, feature_columns, label_offset)

    nw = cfg["training"].get("num_workers", 4)
    bs = engine.get_batch_size(cfg)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              collate_fn=engine.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True, drop_last=False,
                            collate_fn=engine.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True, drop_last=False,
                             collate_fn=engine.collate_fn)

    # Load pretrained checkpoint, infer architecture if needed
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)

    ckpt_channels = infer_in_channels_from_checkpoint(state, args.model)
    if ckpt_channels is not None and ckpt_channels != in_channels:
        print(f"Checkpoint in_channels={ckpt_channels} differs from config ({in_channels}). Using checkpoint value.")
        in_channels = ckpt_channels

    model = engine.build_model(num_classes, in_channels, model_cfg=mcfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys (will be randomly initialized): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

    # Optimizer with scaled LR
    optimizer = engine.get_optimizer(model, cfg)
    for pg in optimizer.param_groups:
        pg["lr"] *= lr_scale

    scheduler = engine.get_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Fine-tuning loop
    best_val_f1 = 0.0
    best_val_oa = 0.0
    best_val_miou = 0.0

    for epoch in range(ft_epochs):
        # Optional freeze backbone for initial epochs
        if epoch < freeze_epochs:
            for name, param in model.named_parameters():
                if "cls" not in name and "conv9" not in name and "conv2" not in name and "head" not in name:
                    param.requires_grad = False
        elif epoch == freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True

        lr_now = optimizer.param_groups[0]["lr"]
        frozen = "frozen" if epoch < freeze_epochs else "unfrozen"
        print(f"\n**** Epoch {epoch+1} ({epoch+1}/{ft_epochs}) [{frozen}] ****")
        print(f"Learning rate: {lr_now:.6f}")

        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"FT E{epoch+1}/{ft_epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = engine.train_step(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Training mean loss: {avg_loss:.6f}")

        # Validation
        print(f"---- EPOCH {epoch+1:03d} EVALUATION ----")
        model.eval()
        val_preds, val_labels = [], []
        for batch in tqdm(val_loader, desc=f"Val E{epoch+1}/{ft_epochs}"):
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

        save_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
        }
        torch.save(save_state, os.path.join(out_dir, "checkpoints", "latest.pth"))

        if val_m["mF1"] > best_val_f1:
            best_val_f1 = val_m["mF1"]
            best_val_oa = val_m["OA"]
            best_val_miou = val_m["mIoU"]
            torch.save(save_state, os.path.join(out_dir, "checkpoints", "best.pth"))
            print(f"  *** New best val mF1: {100*best_val_f1:.2f}% ***")

    # Final test with best checkpoint
    print("\nLoading best checkpoint for final test...")
    best_path = os.path.join(out_dir, "checkpoints", "best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(test_loader, desc="Final Test"):
        pred_flat, lbl_flat = engine.eval_step(model, batch, device)
        all_preds.append(pred_flat.argmax(1).cpu().numpy())
        all_labels.append(lbl_flat.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    test_m = compute_metrics(preds, labels, num_classes)

    print_results(test_m, class_names, prefix=f"Finetune {source_key}->{target_key}")
    save_results(test_m, class_names,
                 os.path.join(out_dir, "test_results.txt"),
                 header=f"Finetune: {args.source_countries} -> {args.finetune_countries}")

    print(f"Fine-tuning complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
