"""
KPConv (Kernel Point Convolution) for semantic segmentation.
Adapted from EARLy/kpconv — self-contained pure-PyTorch implementation
of rigid KPConv with encoder–decoder U-Net architecture (KPFCNN style).

Input:  (B, C, N) — C=3 (xyz) or more
Output: (B, N, num_classes) logits

Reference:
    Thomas et al., "KPConv: Flexible and Deformable Convolution for Point
    Clouds", ICCV 2019.

Optimisations over the original version
----------------------------------------
Speed
  * farthest_point_sample: in-place torch.minimum (no mask alloc), early
    exit when npoint >= N.
  * knn_self / knn_cross: chunked cdist avoids the (B, N, N) ~512 MB
    distance matrix that the old matmul trick built at full resolution.
  * _nn_upsample: chunked cdist avoids the (B, 4096, 2048) ~256 MB matrix
    in the decoder's last upsample step.
  * KPConvLayer: bmm replaces einsum; BN applied after reshape, not
    transpose (avoids extra allocations in KPConvSimple too).

Quality
  * Encoder levels 2-4 now use cross-KNN: each downsampled centroid
    searches the *full* previous level for neighbours (PointNet++ SA
    style), rather than self-KNN on the already-thinned point set.
    This gives each conv layer access to the richer dense feature map.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import register_engine
from datasets.facade_dataset import BlockDataset


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

# Chunk size for cdist to cap per-call GPU memory
_CDIST_CHUNK = 512


def knn_self(xyz, k):
    """Self k-NN.  xyz: (B, N, 3) → idx: (B, N, k)
    Chunked along the query axis to avoid the (B, N, N) distance matrix."""
    B, N, _ = xyz.shape
    k = min(k, N)
    if N <= _CDIST_CHUNK:
        _, idx = torch.cdist(xyz, xyz).topk(k=k, dim=-1, largest=False)
        return idx
    idx = torch.empty(B, N, k, dtype=torch.long, device=xyz.device)
    for i in range(0, N, _CDIST_CHUNK):
        end = min(i + _CDIST_CHUNK, N)
        _, idx[:, i:end] = (
            torch.cdist(xyz[:, i:end], xyz).topk(k=k, dim=-1, largest=False)
        )
    return idx


def knn_cross(query_xyz, support_xyz, k):
    """Cross k-NN: for each query point find k nearest in support.
    query_xyz: (B, M, 3), support_xyz: (B, N, 3) → idx: (B, M, k)
    Chunked along the query axis to cap memory usage."""
    B, M, _ = query_xyz.shape
    N = support_xyz.shape[1]
    k = min(k, N)
    if M <= _CDIST_CHUNK:
        _, idx = torch.cdist(query_xyz, support_xyz).topk(k=k, dim=-1, largest=False)
        return idx
    idx = torch.empty(B, M, k, dtype=torch.long, device=query_xyz.device)
    for i in range(0, M, _CDIST_CHUNK):
        end = min(i + _CDIST_CHUNK, M)
        _, idx[:, i:end] = (
            torch.cdist(query_xyz[:, i:end], support_xyz)
            .topk(k=k, dim=-1, largest=False)
        )
    return idx


def index_points(points, idx):
    """Gather points by index.  points: (B, N, C), idx: (B, ..., K) → (B, ..., K, C)"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS.  xyz: (B, N, 3) → centroids: (B, npoint)
    Uses in-place torch.minimum to avoid boolean-mask allocations."""
    B, N, _ = xyz.shape
    if npoint >= N:
        # Return all indices (no sampling needed)
        return torch.arange(N, dtype=torch.long, device=xyz.device).unsqueeze(0).expand(B, -1)
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)         # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)                   # (B, N)
        torch.minimum(distance, dist, out=distance)              # in-place: no mask alloc
        farthest = distance.argmax(dim=-1)
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    """Ball query.  (B,N,3), (B,S,3) → idx (B,S,nsample)"""
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    sqrdists = torch.cdist(new_xyz, xyz, p=2.0) ** 2   # (B, S, N)
    group_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[sqrdists > radius * radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    first = group_idx[:, :, 0].view(B, S, 1).expand_as(group_idx)
    group_idx[group_idx == N] = first[group_idx == N]
    return group_idx


# ---------------------------------------------------------------------------
#  Kernel-Point Convolution layer (rigid)
# ---------------------------------------------------------------------------

def _generate_kernel_points(num_kpoints, dim=3):
    """Generate roughly uniform kernel points on a unit sphere + origin."""
    pts = [torch.zeros(1, dim)]   # centre kernel point
    if num_kpoints == 1:
        return torch.cat(pts, 0)
    n = num_kpoints - 1
    golden = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n).float()
    theta = 2 * math.pi * indices / golden
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    pts.append(torch.stack([x, y, z], dim=1))
    return torch.cat(pts, 0)          # (K, 3)


class KPConvLayer(nn.Module):
    """Rigid Kernel Point Convolution."""

    def __init__(self, in_channels, out_channels, num_kpoints=15, radius=0.1,
                 sigma=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kpoints = num_kpoints
        self.radius = radius
        self.sigma = sigma or (radius / 2.5)

        # Kernel points (fixed positions)
        kp = _generate_kernel_points(num_kpoints) * (radius * 0.66)
        self.register_buffer("kernel_points", kp)   # (K, 3)

        # Learnable weights per kernel point
        self.weights = nn.Parameter(
            torch.empty(num_kpoints, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, query_xyz, support_xyz, support_feat, neighbor_idx):
        """
        query_xyz:    (B, M, 3)
        support_xyz:  (B, N, 3)
        support_feat: (B, N, Cin)
        neighbor_idx: (B, M, nsample)
        Returns:      (B, M, Cout)
        """
        B, M, nsample = neighbor_idx.shape
        K   = self.num_kpoints
        Cin = self.in_channels
        Cout = self.out_channels

        # Gather neighbour positions & features
        nbr_xyz  = index_points(support_xyz,  neighbor_idx)   # (B, M, ns, 3)
        nbr_feat = index_points(support_feat, neighbor_idx)   # (B, M, ns, Cin)

        # Relative positions w.r.t. query centroid
        rel_pos = nbr_xyz - query_xyz.unsqueeze(2)             # (B, M, ns, 3)

        # Kernel correlation:  h_k(x) = max(0, 1 - ||x - kp_k|| / sigma)
        diffs   = rel_pos.unsqueeze(3) - self.kernel_points    # (B, M, ns, K, 3)
        sq_dist = (diffs ** 2).sum(-1)                         # (B, M, ns, K)
        h = (1.0 - torch.sqrt(sq_dist + 1e-8) / self.sigma).clamp(min=0)

        # Weighted aggregation per kernel point:  (B, M, K, Cin)
        # h: (B,M,ns,K) → permute → (B,M,K,ns)  @  nbr_feat (B,M,ns,Cin)
        weighted = h.permute(0, 1, 3, 2).matmul(nbr_feat)     # (B, M, K, Cin)

        # Per-kernel linear then sum over K — uses bmm for GPU efficiency:
        #   weighted → (K, B*M, Cin),  weights: (K, Cin, Cout)
        #   bmm → (K, B*M, Cout),  .sum(0) → (B*M, Cout)
        wt  = weighted.permute(2, 0, 1, 3).reshape(K, B * M, Cin)
        out = torch.bmm(wt, self.weights).sum(0)               # (B*M, Cout)
        return out.reshape(B, M, Cout)


# ---------------------------------------------------------------------------
#  Building blocks: KPConvSimple, KPConvResBlock
# ---------------------------------------------------------------------------

class KPConvSimple(nn.Module):
    """KPConv + BN + ReLU."""

    def __init__(self, in_ch, out_ch, num_kpoints=15, radius=0.1):
        super().__init__()
        self.kpconv = KPConvLayer(in_ch, out_ch, num_kpoints, radius)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, q_xyz, s_xyz, s_feat, nbr_idx):
        out = self.kpconv(q_xyz, s_xyz, s_feat, nbr_idx)   # (B, M, C)
        B, M, C = out.shape
        # BN on (B*M, C) avoids two transpose allocations
        return F.relu(self.bn(out.reshape(B * M, C)).reshape(B, M, C), inplace=True)


class KPConvResBlock(nn.Module):
    """ResNet-style block with two KPConv layers + skip."""

    def __init__(self, in_ch, out_ch, num_kpoints=15, radius=0.1):
        super().__init__()
        mid = out_ch // 2 if out_ch >= 4 else out_ch
        self.kpc1 = KPConvLayer(in_ch, mid, num_kpoints, radius)
        self.bn1 = nn.BatchNorm1d(mid)
        self.kpc2 = KPConvLayer(mid, out_ch, num_kpoints, radius)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Linear(in_ch, out_ch, bias=False),
                          nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def _apply_bn(self, bn, x):
        B, M, C = x.shape
        return bn(x.reshape(B * M, C)).reshape(B, M, C)

    def forward(self, q_xyz, s_xyz, s_feat, nbr_idx, fps_idx=None):
        """fps_idx: pre-computed FPS indices (B, M) used for the shortcut
        when q_xyz ≠ s_xyz (avoids a redundant FPS call inside forward)."""
        B, M, _ = q_xyz.shape
        if q_xyz is s_xyz or fps_idx is None:
            sc_feat = s_feat[:, :M]
        else:
            sc_feat = index_points(s_feat, fps_idx)
        res = self.shortcut(sc_feat.reshape(B * M, -1)).reshape(B, M, -1)

        out = self.kpc1(q_xyz, s_xyz, s_feat, nbr_idx)
        out = F.relu(self._apply_bn(self.bn1, out), inplace=True)
        nbr_self = knn_self(q_xyz, min(nbr_idx.shape[2], M))
        out = self.kpc2(q_xyz, q_xyz, out, nbr_self)
        out = self._apply_bn(self.bn2, out)
        return F.relu(out + res, inplace=True)


# ---------------------------------------------------------------------------
#  KPFCNN-style U-Net for semantic segmentation
# ---------------------------------------------------------------------------

class KPConvSemSeg(nn.Module):
    """
    KPConv U-Net for semantic segmentation.
    Input:  (B, C, N)  — C channels (first 3 are xyz), N points
    Output: (B, N, num_classes) logits

    Encoder uses cross-KNN (PointNet++ SA style): at each downsampling
    step the new centroids search the *full* previous-level point set for
    neighbours, giving each conv richer input than self-KNN on the already
    thinned set.
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 first_feat_dim: int = 64, num_kpoints: int = 15,
                 base_radius: float = 0.1, nsample: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.nsample = nsample
        self.base_radius = base_radius

        d = first_feat_dim
        r = base_radius

        # ---- Encoder ----
        self.enc1 = KPConvSimple(in_channels, d,      num_kpoints, r)
        self.enc2 = KPConvSimple(d,           d * 2,  num_kpoints, r * 2)
        self.enc3 = KPConvSimple(d * 2,       d * 4,  num_kpoints, r * 4)
        self.enc4 = KPConvSimple(d * 4,       d * 8,  num_kpoints, r * 8)

        # ---- Decoder (feature propagation / upsampling) ----
        self.dec3 = nn.Sequential(
            nn.Linear(d * 8 + d * 4, d * 4), nn.BatchNorm1d(d * 4), nn.ReLU(True))
        self.dec2 = nn.Sequential(
            nn.Linear(d * 4 + d * 2, d * 2), nn.BatchNorm1d(d * 2), nn.ReLU(True))
        self.dec1 = nn.Sequential(
            nn.Linear(d * 2 + d, d), nn.BatchNorm1d(d), nn.ReLU(True))

        # ---- Head ----
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d, num_classes),
        )

    @staticmethod
    def _nn_upsample(xyz_target, xyz_source, feat_source, k=3):
        """Weighted nearest-neighbour interpolation, chunked to cap memory.
        xyz_target: (B, Nt, 3), xyz_source: (B, Ns, 3), feat_source: (B, Ns, C)
        → (B, Nt, C)"""
        B, Nt, _ = xyz_target.shape
        Ns = xyz_source.shape[1]
        C  = feat_source.shape[-1]
        k  = min(k, Ns)
        out = torch.empty(B, Nt, C, device=xyz_target.device,
                          dtype=feat_source.dtype)
        for i in range(0, Nt, _CDIST_CHUNK):
            end = min(i + _CDIST_CHUNK, Nt)
            dists, idx = (
                torch.cdist(xyz_target[:, i:end], xyz_source)
                .topk(k, dim=-1, largest=False)
            )
            w = 1.0 / (dists + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            gathered = index_points(feat_source, idx)          # (B, chunk, k, C)
            out[:, i:end] = (gathered * w.unsqueeze(-1)).sum(dim=2)
        return out

    def forward(self, x):
        """x: (B, C, N)"""
        B, C, N = x.shape
        x = x.transpose(1, 2).contiguous()                    # (B, N, C)
        xyz0 = x[:, :, :3]
        feat0 = x

        ns = self.nsample

        # ---- Encoder ----
        # Level 1 — full resolution, self-KNN
        nbr1  = knn_self(xyz0, min(ns, N))
        feat1 = self.enc1(xyz0, xyz0, feat0, nbr1)            # (B, N,   d)

        # Level 2 — downsample 2×, cross-KNN into level-1 point cloud
        n2   = max(N // 2, 1)
        idx2 = farthest_point_sample(xyz0, n2)
        xyz2 = index_points(xyz0, idx2)
        nbr2 = knn_cross(xyz2, xyz0, min(ns, N))              # xyz2 → xyz0
        feat2 = self.enc2(xyz2, xyz0, feat1, nbr2)            # (B, n2, 2d)

        # Level 3 — downsample 4×, cross-KNN into level-2 point cloud
        n3   = max(N // 4, 1)
        idx3 = farthest_point_sample(xyz2, n3)
        xyz3 = index_points(xyz2, idx3)
        nbr3 = knn_cross(xyz3, xyz2, min(ns, n2))             # xyz3 → xyz2
        feat3 = self.enc3(xyz3, xyz2, feat2, nbr3)            # (B, n3, 4d)

        # Level 4 — downsample 8×, cross-KNN into level-3 point cloud
        n4   = max(N // 8, 1)
        idx4 = farthest_point_sample(xyz3, n4)
        xyz4 = index_points(xyz3, idx4)
        nbr4 = knn_cross(xyz4, xyz3, min(ns, n3))             # xyz4 → xyz3
        feat4 = self.enc4(xyz4, xyz3, feat3, nbr4)            # (B, n4, 8d)

        # ---- Decoder ----
        up3  = self._nn_upsample(xyz3, xyz4, feat4)
        dec3 = self.dec3(
            torch.cat([up3, feat3], dim=-1).reshape(B * n3, -1)
        ).reshape(B, n3, -1)

        up2  = self._nn_upsample(xyz2, xyz3, dec3)
        dec2 = self.dec2(
            torch.cat([up2, feat2], dim=-1).reshape(B * n2, -1)
        ).reshape(B, n2, -1)

        up1  = self._nn_upsample(xyz0, xyz2, dec2)
        dec1 = self.dec1(
            torch.cat([up1, feat1], dim=-1).reshape(B * N, -1)
        ).reshape(B, N, -1)

        # ---- Head ----
        return self.head(dec1.reshape(B * N, -1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
#  Engine (follows the same protocol as dgcnn / pointnet2)
# ---------------------------------------------------------------------------

@register_engine("kpconv")
class KPConvEngine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        return KPConvSemSeg(
            num_classes=num_classes,
            in_channels=in_channels,
            first_feat_dim=mcfg.get("first_feat_dim", 64),
            num_kpoints=mcfg.get("num_kpoints", 15),
            base_radius=mcfg.get("base_radius", 0.1),
            nsample=mcfg.get("nsample", 32),
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["kpconv"]
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
        mcfg = cfg["models"]["kpconv"]
        return optim.Adam(
            model.parameters(),
            lr=mcfg["lr"],
            weight_decay=mcfg["weight_decay"],
        )

    @staticmethod
    def get_scheduler(optimizer, cfg, **kwargs):
        mcfg = cfg["models"]["kpconv"]
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
        feat   = feat.float().to(device)       # (B, N, C)
        labels = labels.long().to(device)      # (B, N)
        pred   = model(feat.transpose(2, 1))   # (B, N, K)
        num_classes = pred.shape[-1]
        loss = criterion(pred.reshape(-1, num_classes), labels.reshape(-1))
        return loss, pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def eval_step(model, batch, device):
        feat, labels = batch
        feat   = feat.float().to(device)
        labels = labels.long().to(device)
        with torch.no_grad():
            pred = model(feat.transpose(2, 1))
        num_classes = pred.shape[-1]
        return pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["kpconv"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["kpconv"]["batch_size"]
