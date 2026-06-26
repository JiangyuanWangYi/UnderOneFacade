"""
OctFormer engine for semantic segmentation.
Wraps the standalone octformer and provides the standardized interface.
Requires: ocnn.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .. import register_engine
from .model import OctFormerSeg
from datasets.facade_dataset import OctreeDataset, octree_collate_fn


class OctFormerSegWrapper(nn.Module):
    """Wraps OctFormerSeg: handles octree construction from batch dicts."""

    def __init__(self, num_classes, in_channels=3,
                 octree_depth=9, full_depth=2, nempty=True, **kwargs):
        super().__init__()
        import ocnn as _ocnn
        self._ocnn = _ocnn
        self.octree_depth = octree_depth
        self.full_depth = full_depth
        self.nempty = nempty
        self.model = OctFormerSeg(
            in_channels=in_channels,
            out_channels=num_classes,
            **kwargs,
        )

    def forward(self, data_dict):
        ocnn = self._ocnn
        coord = data_dict['coord']
        label = data_dict['label']
        feat = data_dict['feat']
        offset = data_dict['offset']
        device = coord.device

        ends = offset.cpu().tolist()
        starts = [0] + ends[:-1]
        batch_id = torch.zeros(coord.shape[0], 1, dtype=torch.long, device=coord.device)
        for i, (s, e) in enumerate(zip(starts, ends)):
            batch_id[s:e] = i
        nbatch = len(ends)

        point = ocnn.octree.Points(
            points=coord.cpu(),
            labels=label.cpu().to(torch.int32).unsqueeze(1),
            features=feat.cpu(),
            batch_id=batch_id.cpu(),
            batch_size=nbatch,
        )
        octree = ocnn.octree.Octree(self.octree_depth, self.full_depth, batch_size=nbatch)
        octree.build_octree(point)
        octree.construct_all_neigh()
        octree = octree.to(device)

        query_pts = torch.cat([point.points.to(device), point.batch_id.to(device)], dim=1)
        data = octree.features[octree.depth].to(device)
        logits = self.model(data, octree, self.octree_depth, query_pts)
        return {"feat": logits}


@register_engine("octformer")
class OctFormerEngine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        return OctFormerSegWrapper(
            num_classes=num_classes,
            in_channels=in_channels,
            octree_depth=mcfg.get("octree_depth", 9),
            full_depth=mcfg.get("full_depth", 2),
            channels=mcfg.get("channels", [96, 192, 384, 384]),
            num_heads=mcfg.get("num_heads", [6, 12, 24, 24]),
            patch_size=mcfg.get("patch_size", 32),
            dilation=mcfg.get("dilation", 4),
            drop_path=mcfg.get("drop_path", 0.5),
            use_dwconv=False,
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["octformer"]
        return OctreeDataset(
            file_paths,
            num_point=mcfg["num_point"],
            lofg=lofg,
            feature_columns=feature_columns,
            label_offset=label_offset,
        )

    @staticmethod
    def get_optimizer(model, cfg):
        mcfg = cfg["models"]["octformer"]
        return optim.AdamW(
            model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"]
        )

    @staticmethod
    def get_scheduler(optimizer, cfg, **kwargs):
        mcfg = cfg["models"]["octformer"]
        sp = mcfg["scheduler_params"]
        if mcfg.get("scheduler") == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=sp["milestones"], gamma=sp["gamma"]
            )
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sp["T_max"], eta_min=sp["eta_min"]
        )

    @staticmethod
    def collate_fn(batch):
        return octree_collate_fn(batch)

    @staticmethod
    def train_step(model, batch, criterion, device):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        output = model(batch)
        pred = output["feat"] if isinstance(output, dict) else output
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
        pred = output["feat"] if isinstance(output, dict) else output
        labels = batch["label"]
        nc = pred.shape[-1]
        return pred.reshape(-1, nc), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["octformer"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["octformer"]["batch_size"]
