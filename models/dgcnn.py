"""
DGCNN for semantic segmentation.
Adapted from UnderOneFacade/DGCNN/model.py (DGCNN_semseg_zaha)
Input: (B, C, N) -- C=3 (xyz) or more
Output: (B, N, num_classes) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import register_engine
from datasets.facade_dataset import BlockDataset


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNNSemSeg(nn.Module):
    """DGCNN semantic segmentation. Input (B, C, N), output (B, N, K)."""

    def __init__(self, num_classes: int, in_channels: int = 3,
                 k: int = 20, emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = k
        cin = in_channels * 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(cin, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), self.bn5, nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, emb_dims, 1, bias=False), self.bn6, nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(
            nn.Conv1d(emb_dims + 192, 512, 1, bias=False), self.bn7, nn.LeakyReLU(0.2))
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False), self.bn8, nn.LeakyReLU(0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(256, num_classes, 1, bias=False)

    def forward(self, x):
        bs = x.size(0)
        npoint = x.size(2)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.repeat(1, 1, npoint)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = x.transpose(2, 1).contiguous()
        return x


@register_engine("dgcnn")
class DGCNNEngine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        return DGCNNSemSeg(
            num_classes=num_classes,
            in_channels=in_channels,
            k=mcfg.get("k", 20),
            emb_dims=mcfg.get("emb_dims", 1024),
            dropout=mcfg.get("dropout", 0.5),
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["dgcnn"]
        return BlockDataset(
            file_paths,
            num_point=mcfg["num_point"],
            block_size=mcfg["block_size"],
            lofg=lofg,
            feature_columns=feature_columns,
            label_offset=label_offset,
            sample_rate=1.0 if split == "train" else 0.5,
        )

    @staticmethod
    def get_optimizer(model, cfg):
        mcfg = cfg["models"]["dgcnn"]
        return optim.Adam(
            model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"]
        )

    @staticmethod
    def get_scheduler(optimizer, cfg, steps_per_epoch=1):
        mcfg = cfg["models"]["dgcnn"]
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
        pred = model(feat_t)                 # (B, N, K)
        num_classes = pred.shape[-1]
        loss = criterion(pred.reshape(-1, num_classes), labels.reshape(-1))
        return loss, pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def eval_step(model, batch, device):
        feat, labels = batch
        feat = feat.float().to(device)
        labels = labels.long().to(device)
        feat_t = feat.transpose(2, 1)
        with torch.no_grad():
            pred = model(feat_t)
        num_classes = pred.shape[-1]
        return pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["dgcnn"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["dgcnn"]["batch_size"]
