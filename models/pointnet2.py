"""
PointNet++ SSG for semantic segmentation.
Adapted from UnderOneFacade/Pointnet++/pointnet2_4phase.py
Input: (B, C, N) -- C=3 (xyz) or more
Output: (B, num_classes, N) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from . import register_engine
from datasets.facade_dataset import BlockDataset


# ---- Pure-PyTorch PointNet++ ops ----

def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    vs = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs = list(idx.shape); rs[0] = 1
    bi = torch.arange(B, device=points.device).view(vs).repeat(rs)
    return points[bi, idx, :]


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius * radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    B = xyz.shape[0]
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, npoint, 1, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        last = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1).contiguous()
        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        last = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out))
            last = out

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interp = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            w = 1.0 / (dists + 1e-8)
            w = w / w.sum(dim=2, keepdim=True)
            interp = (index_points(points2, idx) * w.unsqueeze(-1)).sum(dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1).contiguous()
            new_pts = torch.cat([points1, interp], dim=-1)
        else:
            new_pts = interp
        new_pts = new_pts.permute(0, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_pts = F.relu(bn(conv(new_pts)))
        return new_pts


class PointNet2SemSeg(nn.Module):
    """PointNet++ SSG semantic segmentation. Input (B, C, N), output (B, K, N)."""

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        extra = in_channels - 3 if in_channels > 3 else 0
        sa1_in = 3 + extra

        self.sa1 = PointNetSetAbstraction(1024, 0.10, 32, sa1_in, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.20, 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.40, 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.80, 32, 256 + 3, [256, 256, 512])

        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        l0_xyz = x[:, :3, :]
        l0_points = x[:, 3:, :] if self.in_channels > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return x


# ---- Engine ----

@register_engine("pointnet2")
class PointNet2Engine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        return PointNet2SemSeg(num_classes=num_classes, in_channels=in_channels)

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["pointnet2"]
        return BlockDataset(
            file_paths,
            num_point=mcfg["num_point"],
            block_size=mcfg["block_size"],
            lofg=lofg,
            feature_columns=feature_columns,
            label_offset=label_offset,
            sample_rate=1.0,
        )

    @staticmethod
    def get_optimizer(model, cfg):
        mcfg = cfg["models"]["pointnet2"]
        return optim.Adam(
            model.parameters(),
            lr=mcfg["lr"],
            weight_decay=mcfg["weight_decay"],
        )

    @staticmethod
    def get_scheduler(optimizer, cfg, steps_per_epoch=1):
        mcfg = cfg["models"]["pointnet2"]
        sp = mcfg["scheduler_params"]
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sp["T_max"], eta_min=sp["eta_min"]
        )

    @staticmethod
    def collate_fn(batch):
        return torch.utils.data.dataloader.default_collate(batch)

    @staticmethod
    def train_step(model, batch, criterion, device):
        feat, labels = batch
        feat = feat.float().to(device)       # (B, N, C)
        labels = labels.long().to(device)    # (B, N)
        feat_t = feat.transpose(2, 1)       # (B, C, N)
        pred = model(feat_t)                 # (B, K, N)
        num_classes = pred.shape[1]
        loss = criterion(
            pred.permute(0, 2, 1).reshape(-1, num_classes),
            labels.reshape(-1),
        )
        return loss, pred.permute(0, 2, 1).reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def eval_step(model, batch, device):
        feat, labels = batch
        feat = feat.float().to(device)
        labels = labels.long().to(device)
        feat_t = feat.transpose(2, 1)
        with torch.no_grad():
            pred = model(feat_t)
        num_classes = pred.shape[1]
        return pred.permute(0, 2, 1).reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["pointnet2"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["pointnet2"]["batch_size"]
