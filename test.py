#!/usr/bin/env python
"""
Zero-shot test: load pre-trained weights, evaluate on target country.
Usage:
    python test.py --model pointnet2 --lofg lofg2 --features xyz \
        --weights auto --source_countries nottingham \
        --test_countries singapore --config config.yaml
"""

import os
import argparse
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
    p = argparse.ArgumentParser("Cross-continent zero-shot test")
    p.add_argument("--model", required=True)
    p.add_argument("--lofg", required=True, choices=["lofg2", "lofg3"])
    p.add_argument("--features", default="xyz", choices=["xyz", "rgbi"])
    p.add_argument("--weights", default="auto", help="Path to weights or 'auto'")
    p.add_argument("--source_countries", required=True, help="Source countries (for auto weight resolution)")
    p.add_argument("--test_countries", required=True, help="Target countries to test on")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="auto")
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

    # Resolve weights
    if args.weights == "auto":
        weights_path = resolve_weight_path(
            cfg["weights"]["root"], args.source_countries, args.model, args.lofg, args.features
        )
    else:
        weights_path = args.weights

    if not os.path.exists(weights_path):
        print(f"ERROR: weights not found: {weights_path}")
        return

    source_key = "_".join(sorted(args.source_countries.split(",")))
    target_key = "_".join(sorted(args.test_countries.split(",")))
    run_name = f"zeroshot_{args.model}_{args.lofg}_{args.features}_{source_key}_to_{target_key}"

    print(f"Model: {args.model} | LoFG: {args.lofg} | Features: {args.features}")
    print(f"Source: {args.source_countries} | Target: {args.test_countries}")
    print(f"Weights: {weights_path}")

    # Load checkpoint and infer in_channels FIRST before building dataset
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = extract_state_dict(ckpt)

    ckpt_channels = infer_in_channels_from_checkpoint(state, args.model)
    if ckpt_channels is not None and ckpt_channels != in_channels:
        print(f"Checkpoint in_channels={ckpt_channels} differs from config ({in_channels}). Using checkpoint value.")
        in_channels = ckpt_channels

    # Adjust feature columns if checkpoint expects different channels
    if in_channels != feat_info["num_channels"]:
        if in_channels == 6 and feat_info["num_channels"] == 3:
            feature_columns = [0, 1, 2, 0, 1, 2]  # duplicate xyz as features

    # Data
    test_dirs = get_data_paths(cfg, args.test_countries, "test")
    test_files = discover_files(test_dirs)
    print(f"Test files: {len(test_files)}")

    test_ds = engine.build_dataset(test_files, cfg, "test", args.lofg, feature_columns, label_offset)
    bs = engine.get_batch_size(cfg)
    nw = cfg["training"].get("num_workers", 4)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True, drop_last=False,
                             collate_fn=engine.collate_fn)

    model = engine.build_model(num_classes, in_channels, model_cfg=mcfg).to(device)
    model.load_state_dict(state, strict=False)
    print("Weights loaded.")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(test_loader, desc="Testing"):
        pred_flat, lbl_flat = engine.eval_step(model, batch, device)
        all_preds.append(pred_flat.argmax(1).cpu().numpy())
        all_labels.append(lbl_flat.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(preds, labels, num_classes)

    print_results(metrics, class_names, prefix=run_name)

    # Save
    out_dir = os.path.join(cfg["output"]["root"], "test", run_name)
    save_results(metrics, class_names,
                 os.path.join(out_dir, "test_results.txt"),
                 header=f"Zero-shot: {args.source_countries} -> {args.test_countries}")

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
