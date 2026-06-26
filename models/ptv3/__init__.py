"""PTv3 engine for cross-continent facade segmentation."""

import torch
import torch.nn as nn
import torch.optim as optim

from .. import register_engine
from datasets.facade_dataset import SparseVoxelDataset, sparse_collate_fn

import flash_attn
from .model import PointTransformerV3


class PTv3SegHead(nn.Module):

    def __init__(self, num_classes, in_channels=3, **kwargs):
        super().__init__()
        mcfg = kwargs.pop("model_cfg", {})
        use_flash = (flash_attn is not None) and mcfg.get("enable_flash", True)
        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            enable_flash=use_flash,
            **kwargs,
        )
        self.cls = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, data_dict):
        point = self.backbone(data_dict)
        logits = self.cls(point.feat)
        return {"feat": logits}


@register_engine("ptv3")
class PTv3Engine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        return PTv3SegHead(
            num_classes=num_classes,
            in_channels=in_channels,
            model_cfg=mcfg,
            order=mcfg.get("order", ["z", "z-trans", "hilbert", "hilbert-trans"]),
            enc_depths=mcfg.get("enc_depths", [2, 2, 2, 6, 2]),
            enc_channels=mcfg.get("enc_channels", [32, 64, 128, 256, 512]),
            enc_num_head=mcfg.get("enc_num_head", [2, 4, 8, 16, 32]),
            dec_depths=mcfg.get("dec_depths", [2, 2, 2, 2]),
            dec_channels=mcfg.get("dec_channels", [64, 64, 128, 256]),
            dec_num_head=mcfg.get("dec_num_head", [4, 4, 8, 16]),
            enc_patch_size=tuple([mcfg.get("patch_size", 1024)] * 5),
            dec_patch_size=tuple([mcfg.get("patch_size", 1024)] * 4),
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["ptv3"]
        return SparseVoxelDataset(
            file_paths,
            voxel_size=mcfg["voxel_size"],
            num_point=mcfg["num_point"],
            lofg=lofg,
            feature_columns=feature_columns,
            label_offset=label_offset,
        )

    @staticmethod
    def get_optimizer(model, cfg):
        mcfg = cfg["models"]["ptv3"]
        if mcfg.get("optimizer", "AdamW") == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=mcfg["lr"],
                momentum=mcfg.get("momentum", 0.9),
                weight_decay=mcfg["weight_decay"],
            )
        return optim.AdamW(
            model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"]
        )

    @staticmethod
    def get_scheduler(optimizer, cfg, **kwargs):
        mcfg = cfg["models"]["ptv3"]
        sp = mcfg["scheduler_params"]
        if mcfg.get("scheduler") == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=sp["milestones"], gamma=sp["gamma"]
            )
        epochs = mcfg["epochs"]
        warmup = sp.get("warmup_epochs", 10)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=sp["eta_min"]
        )
        linear_warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, [linear_warmup, cosine], milestones=[warmup]
        )

    @staticmethod
    def collate_fn(batch):
        return sparse_collate_fn(batch)

    @staticmethod
    def train_step(model, batch, criterion, device):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        output = model(batch)
        pred = output.get("feat", output.get("seg_logits")) if isinstance(output, dict) else (output.feat if hasattr(output, "feat") else output)
        labels = batch["label"]
        nc = pred.shape[-1]
        loss = criterion(pred.reshape(-1, nc), labels.reshape(-1))
        return loss, pred.reshape(-1, nc), labels.reshape(-1)

    @staticmethod
    def eval_step(model, batch, device):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            output = model(batch)
        pred = output.get("feat", output.get("seg_logits")) if isinstance(output, dict) else (output.feat if hasattr(output, "feat") else output)
        labels = batch["label"]
        nc = pred.shape[-1]
        return pred.reshape(-1, nc), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["ptv3"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["ptv3"]["batch_size"]
