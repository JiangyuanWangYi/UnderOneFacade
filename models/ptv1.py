"""
Point Transformer V1 for semantic segmentation.
Matches EARLy/point_transformer/PT_LentonRd.ipynb architecture.
Input: [coords (N,3), feat (N,C), offset (B,)]
Output: logits (N, num_classes)

Requires pointops CUDA library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import register_engine
from datasets.facade_dataset import PointOffsetDataset, point_collate_fn
from pointops import query_and_group as queryandgroup
from pointops import farthest_point_sampling as furthestsampling
from pointops import interpolation


class LayerNorm1d(nn.Module):
    """LayerNorm that works on (N, C) or (N, L, C) tensors — matches official PTv1."""
    def __init__(self, num_channels: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid = mid = out_planes // 1
        self.out = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid)
        self.linear_k = nn.Linear(in_planes, mid)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3), LayerNorm1d(3), nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, mid // share_planes),
            LayerNorm1d(mid // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo
        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)
        x_k, _ = queryandgroup(self.nsample, p, p, x_k.contiguous(), None, o, o, with_xyz=True)
        torch.cuda.synchronize()
        x_v, _ = queryandgroup(self.nsample, p, p, x_v.contiguous(), None, o, o, with_xyz=False)
        torch.cuda.synchronize()
        p_r = x_k[:, :, 0:3]
        x_k = x_k[:, :, 3:]
        # LayerNorm1d handles (n, nsample, C) directly — no flatten/reshape needed
        p_r = self.linear_p(p_r)                   # (n, nsample, out_planes)
        r_qk = x_k - x_q.unsqueeze(1) + p_r
        w = self.linear_w(r_qk)                    # (n, nsample, out_planes // share_planes)
        w = self.softmax(w)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride = stride
        self.nsample = nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                cnt = (o[i].item() - o[i - 1].item()) // self.stride
                count += cnt
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.long, device=p.device)
            # Validate inputs before CUDA ops
            assert p.shape[0] > 0, f"Empty points tensor: {p.shape}"
            assert n_o[-1] > 0, f"Invalid n_o: {n_o}"
            assert n_o[-1] <= p.shape[0], f"n_o exceeds points: {n_o[-1]} > {p.shape[0]}"
            idx = furthestsampling(p, o, n_o)
            torch.cuda.synchronize()
            n_p = p[idx.long(), :]
            x, _ = queryandgroup(self.nsample, p, n_p, x.contiguous(), None, o, n_o, with_xyz=True)
            torch.cuda.synchronize()
            x = self.relu(self.bn(self.linear(x.view(-1, x.shape[-1])).view(
                x.shape[0], self.nsample, -1).permute(0, 2, 1).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                s_i = 0 if i == 0 else o[i - 1]
                e_i = o[i]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.mean(0, True).expand_as(x_b))), 1)
                x_tmp.append(x_b)
            x = self.linear1(torch.cat(x_tmp, 0))
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    """Point Transformer V1 segmentation -- matches EARLy/point_transformer."""

    def __init__(self, num_classes, in_channels=3, blocks=None):
        super().__init__()
        if blocks is None:
            blocks = [1, 2, 2, 2, 2]  # matches official PTv1 (EARLy)
        self.c = in_channels

        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        # enc1 input dim: xyz-only (3) -> in_channels; rgbi (7) etc. -> cat(p0,x0) = 3+in_channels
        self.in_planes = (3 + in_channels) if in_channels != 3 else in_channels
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride[0], nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride[1], nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride[2], nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride[3], nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride[4], nsample[4])

        self.dec5 = self._make_dec(planes[4], 1, share_planes, nsample[4], is_head=True)
        self.dec4 = self._make_dec(planes[3], 1, share_planes, nsample[3])
        self.dec3 = self._make_dec(planes[2], 1, share_planes, nsample[2])
        self.dec2 = self._make_dec(planes[1], 1, share_planes, nsample[1])
        self.dec1 = self._make_dec(planes[0], 1, share_planes, nsample[0])

        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = [
            TransitionDown(self.in_planes, planes * PointTransformerBlock.expansion, stride, nsample)
        ]
        self.in_planes = planes * PointTransformerBlock.expansion
        for _ in range(blocks):  # range(blocks) matches EARLy; blocks=[1,2,2,2,2]
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = [
            TransitionUp(self.in_planes, None if is_head else planes * PointTransformerBlock.expansion)
        ]
        self.in_planes = planes * PointTransformerBlock.expansion
        for _ in range(blocks):  # range(blocks) matches EARLy; decoder blocks=1
            layers.append(PointTransformerBlock(self.in_planes, self.in_planes, share_planes, nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]

        return self.cls(x1)


@register_engine("ptv1")
class PTv1Engine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        blocks = mcfg.get("blocks", [1, 2, 2, 2, 2])
        return PointTransformerSeg(
            num_classes=num_classes, in_channels=in_channels, blocks=blocks,
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["ptv1"]
        # repeat: sample each scene N times per epoch (training only); val uses repeat=1
        repeat = mcfg.get("repeat", 1) if split == "train" else 1
        return PointOffsetDataset(
            file_paths,
            num_point=mcfg["num_point"],
            lofg=lofg,
            feature_columns=feature_columns,
            label_offset=label_offset,
            repeat=repeat,
        )

    @staticmethod
    def get_optimizer(model, cfg):
        mcfg = cfg["models"]["ptv1"]
        opt_type = mcfg.get("optimizer", "AdamW")
        if opt_type == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=mcfg["lr"],
                momentum=mcfg.get("momentum", 0.9),
                weight_decay=mcfg["weight_decay"],
            )
        else:  # AdamW
            return optim.AdamW(
                model.parameters(),
                lr=mcfg["lr"],
                weight_decay=mcfg["weight_decay"],
            )

    @staticmethod
    def get_scheduler(optimizer, cfg, steps_per_epoch=1):
        mcfg = cfg["models"]["ptv1"]
        sp = mcfg["scheduler_params"]
        epochs = mcfg["epochs"]
        sched_type = mcfg.get("scheduler", "multistep")
        if sched_type == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sp["max_lr"],
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=sp.get("pct_start", 0.05),
                div_factor=sp.get("div_factor", 10.0),
                final_div_factor=sp.get("final_div_factor", 1000.0),
                anneal_strategy="cos",
            )
        else:  # multistep
            milestones = [int(p * epochs) for p in sp["milestones_pct"]]
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=sp["gamma"]
            )

    @staticmethod
    def collate_fn(batch):
        return point_collate_fn(batch)

    @staticmethod
    def train_step(model, batch, criterion, device):
        coords, feat, labels, offset = batch
        coords = coords.to(device)
        feat = feat.to(device)
        labels = labels.to(device)
        offset = offset.to(device)
        pred = model([coords, feat, offset])
        loss = criterion(pred, labels)
        return loss, pred, labels

    @staticmethod
    def eval_step(model, batch, device):
        coords, feat, labels, offset = batch
        coords = coords.to(device)
        feat = feat.to(device)
        labels = labels.to(device)
        offset = offset.to(device)
        with torch.no_grad():
            pred = model([coords, feat, offset])
        return pred, labels

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["ptv1"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["ptv1"]["batch_size"]
