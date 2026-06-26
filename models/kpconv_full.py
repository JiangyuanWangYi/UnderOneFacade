"""
KPConv-Full: Full-featured Kernel Point Convolution for semantic segmentation.

This is an enhanced version with:
  - Deformable KPConv (learns kernel point offsets)
  - Deeper encoder with ResNet-style blocks (similar to original KPFCNN)
  - Configurable kernel influence: 'constant', 'linear', 'gaussian'
  - Modulated convolutions option
  - Deformation regularization losses

Input:  (B, C, N) — C=3 (xyz) or more
Output: (B, N, num_classes) logits

Reference:
    Thomas et al., "KPConv: Flexible and Deformable Convolution for Point
    Clouds", ICCV 2019.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import register_engine
from datasets.facade_dataset import BlockDataset


_CDIST_CHUNK = 512


# ---------------------------------------------------------------------------
#  Utility helpers (same as kpconv.py)
# ---------------------------------------------------------------------------

def knn_self(xyz, k):
    """Self k-NN. xyz: (B, N, 3) → idx: (B, N, k)"""
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
    """Cross k-NN: for each query point find k nearest in support."""
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
    """Gather points by index. points: (B, N, C), idx: (B, ..., K) → (B, ..., K, C)"""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS. xyz: (B, N, 3) → centroids: (B, npoint)"""
    B, N, _ = xyz.shape
    if npoint >= N:
        return torch.arange(N, dtype=torch.long, device=xyz.device).unsqueeze(0).expand(B, -1)
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        torch.minimum(distance, dist, out=distance)
        farthest = distance.argmax(dim=-1)
    return centroids


# ---------------------------------------------------------------------------
#  Kernel Point Generation
# ---------------------------------------------------------------------------

def _generate_kernel_points(num_kpoints, dim=3):
    """Generate roughly uniform kernel points on a unit sphere + origin."""
    pts = [torch.zeros(1, dim)]
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
    return torch.cat(pts, 0)


# ---------------------------------------------------------------------------
#  Kernel Influence Functions
# ---------------------------------------------------------------------------

def kernel_influence_constant(sq_dist, sigma):
    """Constant influence within sigma radius."""
    return (sq_dist < sigma * sigma).float()


def kernel_influence_linear(sq_dist, sigma):
    """Linear decay: h = max(0, 1 - dist/sigma)"""
    dist = torch.sqrt(sq_dist + 1e-8)
    return (1.0 - dist / sigma).clamp(min=0)


def kernel_influence_gaussian(sq_dist, sigma):
    """Gaussian influence: h = exp(-dist^2 / (2*sigma^2))"""
    return torch.exp(-sq_dist / (2 * sigma * sigma))


INFLUENCE_FN = {
    'constant': kernel_influence_constant,
    'linear': kernel_influence_linear,
    'gaussian': kernel_influence_gaussian,
}


# ---------------------------------------------------------------------------
#  Rigid KPConv Layer
# ---------------------------------------------------------------------------

class KPConvLayer(nn.Module):
    """Rigid Kernel Point Convolution with configurable influence."""

    def __init__(self, in_channels, out_channels, num_kpoints=15, radius=0.1,
                 sigma=None, influence='linear', aggregation='sum'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kpoints = num_kpoints
        self.radius = radius
        self.sigma = sigma or (radius / 2.5)
        self.influence = influence
        self.aggregation = aggregation

        kp = _generate_kernel_points(num_kpoints) * (radius * 0.66)
        self.register_buffer("kernel_points", kp)

        self.weights = nn.Parameter(torch.empty(num_kpoints, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, query_xyz, support_xyz, support_feat, neighbor_idx):
        """
        query_xyz:    (B, M, 3)
        support_xyz:  (B, N, 3)
        support_feat: (B, N, Cin)
        neighbor_idx: (B, M, nsample)
        Returns:      (B, M, Cout)
        """
        B, M, _ = neighbor_idx.shape
        K = self.num_kpoints

        nbr_xyz = index_points(support_xyz, neighbor_idx)
        nbr_feat = index_points(support_feat, neighbor_idx)
        rel_pos = nbr_xyz - query_xyz.unsqueeze(2)

        diffs = rel_pos.unsqueeze(3) - self.kernel_points
        sq_dist = (diffs ** 2).sum(-1)

        influence_fn = INFLUENCE_FN[self.influence]
        h = influence_fn(sq_dist, self.sigma)

        if self.aggregation == 'closest':
            closest_idx = sq_dist.argmin(dim=2, keepdim=True)
            h = torch.zeros_like(h).scatter_(2, closest_idx.expand(-1, -1, -1, K), 1.0)

        weighted = h.permute(0, 1, 3, 2).matmul(nbr_feat)
        wt = weighted.permute(2, 0, 1, 3).reshape(K, B * M, self.in_channels)
        out = torch.bmm(wt, self.weights).sum(0)
        return out.reshape(B, M, self.out_channels)


# ---------------------------------------------------------------------------
#  Deformable KPConv Layer
# ---------------------------------------------------------------------------

class KPConvDeformableLayer(nn.Module):
    """
    Deformable Kernel Point Convolution.
    Learns offsets for each kernel point based on input features.
    """

    def __init__(self, in_channels, out_channels, num_kpoints=15, radius=0.1,
                 sigma=None, influence='linear', aggregation='sum', modulated=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kpoints = num_kpoints
        self.radius = radius
        self.sigma = sigma or (radius / 2.5)
        self.influence = influence
        self.aggregation = aggregation
        self.modulated = modulated

        kp = _generate_kernel_points(num_kpoints) * (radius * 0.66)
        self.register_buffer("kernel_points", kp)

        self.weights = nn.Parameter(torch.empty(num_kpoints, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # Offset prediction: predicts (K, 3) offsets from aggregated neighbor features
        self.offset_mlp = nn.Sequential(
            nn.Linear(in_channels, num_kpoints * 3),
        )
        nn.init.zeros_(self.offset_mlp[0].weight)
        nn.init.zeros_(self.offset_mlp[0].bias)

        # Optional modulation (per-kernel-point scaling)
        if modulated:
            self.modulation_mlp = nn.Sequential(
                nn.Linear(in_channels, num_kpoints),
                nn.Sigmoid(),
            )

        # Store deformed points for regularization loss
        self.deformed_kp = None

    def forward(self, query_xyz, support_xyz, support_feat, neighbor_idx):
        B, M, _ = neighbor_idx.shape
        K = self.num_kpoints

        nbr_xyz = index_points(support_xyz, neighbor_idx)
        nbr_feat = index_points(support_feat, neighbor_idx)
        rel_pos = nbr_xyz - query_xyz.unsqueeze(2)

        # Compute offsets from mean neighbor features
        mean_feat = nbr_feat.mean(dim=2)  # (B, M, Cin)
        offsets = self.offset_mlp(mean_feat).reshape(B, M, K, 3)

        # Deformed kernel points
        deformed_kp = self.kernel_points.unsqueeze(0).unsqueeze(0) + offsets  # (B, M, K, 3)
        self.deformed_kp = deformed_kp

        # Compute influence with deformed kernel points
        diffs = rel_pos.unsqueeze(3) - deformed_kp.unsqueeze(2)  # (B, M, ns, K, 3)
        sq_dist = (diffs ** 2).sum(-1)

        influence_fn = INFLUENCE_FN[self.influence]
        h = influence_fn(sq_dist, self.sigma)

        if self.aggregation == 'closest':
            closest_idx = sq_dist.argmin(dim=2, keepdim=True)
            h = torch.zeros_like(h).scatter_(2, closest_idx.expand(-1, -1, -1, K), 1.0)

        weighted = h.permute(0, 1, 3, 2).matmul(nbr_feat)  # (B, M, K, Cin)

        # Apply modulation if enabled
        if self.modulated:
            mod = self.modulation_mlp(mean_feat)  # (B, M, K)
            weighted = weighted * mod.unsqueeze(-1)

        wt = weighted.permute(2, 0, 1, 3).reshape(K, B * M, self.in_channels)
        out = torch.bmm(wt, self.weights).sum(0)
        return out.reshape(B, M, self.out_channels)


# ---------------------------------------------------------------------------
#  Building Blocks
# ---------------------------------------------------------------------------

def _apply_bn(bn, x):
    """Apply BatchNorm to (B, M, C) tensor."""
    B, M, C = x.shape
    return bn(x.reshape(B * M, C)).reshape(B, M, C)


class KPConvBlock(nn.Module):
    """KPConv + BN + ReLU (rigid or deformable)."""

    def __init__(self, in_ch, out_ch, num_kpoints=15, radius=0.1,
                 influence='linear', deformable=False, modulated=False):
        super().__init__()
        if deformable:
            self.conv = KPConvDeformableLayer(
                in_ch, out_ch, num_kpoints, radius,
                influence=influence, modulated=modulated
            )
        else:
            self.conv = KPConvLayer(in_ch, out_ch, num_kpoints, radius, influence=influence)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, q_xyz, s_xyz, s_feat, nbr_idx):
        out = self.conv(q_xyz, s_xyz, s_feat, nbr_idx)
        return F.relu(_apply_bn(self.bn, out), inplace=True)


class KPConvResBlock(nn.Module):
    """ResNet-style block with two KPConv layers + skip (rigid or deformable)."""

    def __init__(self, in_ch, out_ch, num_kpoints=15, radius=0.1,
                 influence='linear', deformable=False, modulated=False, strided=False):
        super().__init__()
        self.strided = strided
        mid = out_ch // 4 if out_ch >= 16 else out_ch

        # First conv: bottleneck down
        if deformable:
            self.conv1 = KPConvDeformableLayer(
                in_ch, mid, num_kpoints, radius, influence=influence, modulated=modulated
            )
        else:
            self.conv1 = KPConvLayer(in_ch, mid, num_kpoints, radius, influence=influence)
        self.bn1 = nn.BatchNorm1d(mid)

        # Second conv: same resolution
        if deformable:
            self.conv2 = KPConvDeformableLayer(
                mid, mid, num_kpoints, radius, influence=influence, modulated=modulated
            )
        else:
            self.conv2 = KPConvLayer(mid, mid, num_kpoints, radius, influence=influence)
        self.bn2 = nn.BatchNorm1d(mid)

        # Third conv: bottleneck up
        if deformable:
            self.conv3 = KPConvDeformableLayer(
                mid, out_ch, num_kpoints, radius, influence=influence, modulated=modulated
            )
        else:
            self.conv3 = KPConvLayer(mid, out_ch, num_kpoints, radius, influence=influence)
        self.bn3 = nn.BatchNorm1d(out_ch)

        # Shortcut
        self.shortcut = (
            nn.Sequential(nn.Linear(in_ch, out_ch, bias=False), nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, q_xyz, s_xyz, s_feat, nbr_idx, fps_idx=None):
        B, M, _ = q_xyz.shape

        # Shortcut
        if self.strided and fps_idx is not None:
            sc_feat = index_points(s_feat, fps_idx)
        else:
            sc_feat = s_feat[:, :M] if s_feat.shape[1] >= M else s_feat
        res = self.shortcut(sc_feat.reshape(B * M, -1)).reshape(B, M, -1)

        # Main path
        out = self.conv1(q_xyz, s_xyz, s_feat, nbr_idx)
        out = F.relu(_apply_bn(self.bn1, out), inplace=True)

        nbr_self = knn_self(q_xyz, min(nbr_idx.shape[2], M))
        out = self.conv2(q_xyz, q_xyz, out, nbr_self)
        out = F.relu(_apply_bn(self.bn2, out), inplace=True)

        out = self.conv3(q_xyz, q_xyz, out, nbr_self)
        out = _apply_bn(self.bn3, out)

        return F.relu(out + res, inplace=True)


class UnaryBlock(nn.Module):
    """Simple Linear + BN + ReLU for decoder."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        B, N, _ = x.shape
        out = self.mlp(x.reshape(B * N, -1))
        out = F.relu(self.bn(out), inplace=True)
        return out.reshape(B, N, -1)


# ---------------------------------------------------------------------------
#  KPFCNN-Full: Deep U-Net with Deformable KPConv
# ---------------------------------------------------------------------------

class KPConvFullSemSeg(nn.Module):
    """
    Full-featured KPConv U-Net for semantic segmentation.

    Architecture (similar to original KPFCNN):
      Encoder: simple → resnetb → resnetb_strided → resnetb → resnetb →
               resnetb_strided → resnetb → resnetb → resnetb_strided →
               resnetb_deformable → resnetb_deformable → resnetb_deformable_strided →
               resnetb_deformable → resnetb_deformable
      Decoder: nearest_upsample + unary (×4)

    Input:  (B, C, N) — C channels (first 3 are xyz)
    Output: (B, N, num_classes) logits
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 first_feat_dim: int = 128, num_kpoints: int = 15,
                 base_radius: float = 0.1, nsample: int = 32,
                 influence: str = 'linear', modulated: bool = False,
                 deform_fitting_power: float = 1.0, repulse_extent: float = 1.2):
        super().__init__()
        self.in_channels = in_channels
        self.nsample = nsample
        self.base_radius = base_radius
        self.num_kpoints = num_kpoints
        self.influence = influence
        self.modulated = modulated
        self.deform_fitting_power = deform_fitting_power
        self.repulse_extent = repulse_extent

        d = first_feat_dim
        r = base_radius

        # ---- Encoder ----
        # Level 1: full resolution
        self.enc1_simple = KPConvBlock(in_channels, d, num_kpoints, r, influence)
        self.enc1_res1 = KPConvResBlock(d, d, num_kpoints, r, influence)

        # Level 2: downsample 2x
        self.enc2_res1 = KPConvResBlock(d, d * 2, num_kpoints, r * 2, influence, strided=True)
        self.enc2_res2 = KPConvResBlock(d * 2, d * 2, num_kpoints, r * 2, influence)

        # Level 3: downsample 4x
        self.enc3_res1 = KPConvResBlock(d * 2, d * 4, num_kpoints, r * 4, influence, strided=True)
        self.enc3_res2 = KPConvResBlock(d * 4, d * 4, num_kpoints, r * 4, influence)

        # Level 4: downsample 8x (start deformable)
        self.enc4_res1 = KPConvResBlock(d * 4, d * 8, num_kpoints, r * 8, influence,
                                        deformable=True, modulated=modulated, strided=True)
        self.enc4_res2 = KPConvResBlock(d * 8, d * 8, num_kpoints, r * 8, influence,
                                        deformable=True, modulated=modulated)

        # Level 5: downsample 16x (deformable)
        self.enc5_res1 = KPConvResBlock(d * 8, d * 16, num_kpoints, r * 16, influence,
                                        deformable=True, modulated=modulated, strided=True)
        self.enc5_res2 = KPConvResBlock(d * 16, d * 16, num_kpoints, r * 16, influence,
                                        deformable=True, modulated=modulated)

        # ---- Decoder ----
        self.dec4 = UnaryBlock(d * 16 + d * 8, d * 8)
        self.dec3 = UnaryBlock(d * 8 + d * 4, d * 4)
        self.dec2 = UnaryBlock(d * 4 + d * 2, d * 2)
        self.dec1 = UnaryBlock(d * 2 + d, d)

        # ---- Head ----
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(d, num_classes),
        )

        # Store deformable layers for regularization
        self._deformable_layers = [
            self.enc4_res1.conv1, self.enc4_res1.conv2, self.enc4_res1.conv3,
            self.enc4_res2.conv1, self.enc4_res2.conv2, self.enc4_res2.conv3,
            self.enc5_res1.conv1, self.enc5_res1.conv2, self.enc5_res1.conv3,
            self.enc5_res2.conv1, self.enc5_res2.conv2, self.enc5_res2.conv3,
        ]

    @staticmethod
    def _nn_upsample(xyz_target, xyz_source, feat_source, k=3):
        """Weighted nearest-neighbour interpolation."""
        B, Nt, _ = xyz_target.shape
        Ns = xyz_source.shape[1]
        C = feat_source.shape[-1]
        k = min(k, Ns)
        out = torch.empty(B, Nt, C, device=xyz_target.device, dtype=feat_source.dtype)
        for i in range(0, Nt, _CDIST_CHUNK):
            end = min(i + _CDIST_CHUNK, Nt)
            dists, idx = (
                torch.cdist(xyz_target[:, i:end], xyz_source)
                .topk(k, dim=-1, largest=False)
            )
            w = 1.0 / (dists + 1e-8)
            w = w / w.sum(dim=-1, keepdim=True)
            gathered = index_points(feat_source, idx)
            out[:, i:end] = (gathered * w.unsqueeze(-1)).sum(dim=2)
        return out

    def get_deformation_loss(self, fitting_power=None, repulse_extent=None):
        """
        Compute deformation regularization loss.
        - Fitting loss: penalizes kernel points far from input points
        - Repulsive loss: prevents kernel points from collapsing together
        """
        fitting_power = fitting_power if fitting_power is not None else self.deform_fitting_power
        repulse_extent = repulse_extent if repulse_extent is not None else self.repulse_extent
        fitting_loss = 0.0
        repulsive_loss = 0.0
        count = 0

        for layer in self._deformable_layers:
            if hasattr(layer, 'deformed_kp') and layer.deformed_kp is not None:
                kp = layer.deformed_kp  # (B, M, K, 3)
                B, M, K, _ = kp.shape

                # Repulsive loss: pairwise distances between kernel points
                kp_flat = kp.reshape(B * M, K, 3)
                kp_dists = torch.cdist(kp_flat, kp_flat)  # (B*M, K, K)
                # Exclude diagonal
                mask = ~torch.eye(K, dtype=torch.bool, device=kp.device)
                kp_dists = kp_dists[:, mask].reshape(B * M, K, K - 1)
                # Repulsion: penalize if closer than repulse_extent * sigma
                sigma = layer.sigma
                repulse_thresh = repulse_extent * sigma
                repulse = F.relu(repulse_thresh - kp_dists)
                repulsive_loss += repulse.mean()

                # Fitting loss: distance from origin (kernel points should stay near original positions)
                original_kp = layer.kernel_points  # (K, 3)
                offsets = kp - original_kp.unsqueeze(0).unsqueeze(0)
                fitting_loss += (offsets ** 2).sum(-1).mean()

                count += 1

        if count > 0:
            fitting_loss /= count
            repulsive_loss /= count

        return fitting_power * (fitting_loss + repulsive_loss)

    def forward(self, x):
        """x: (B, C, N)"""
        B, _, N = x.shape
        x = x.transpose(1, 2).contiguous()
        xyz0 = x[:, :, :3]
        feat0 = x

        ns = self.nsample

        # ---- Encoder Level 1 ----
        nbr1 = knn_self(xyz0, min(ns, N))
        feat1 = self.enc1_simple(xyz0, xyz0, feat0, nbr1)
        feat1 = self.enc1_res1(xyz0, xyz0, feat1, nbr1)

        # ---- Encoder Level 2 ----
        n2 = max(N // 2, 1)
        idx2 = farthest_point_sample(xyz0, n2)
        xyz2 = index_points(xyz0, idx2)
        nbr2 = knn_cross(xyz2, xyz0, min(ns, N))
        feat2 = self.enc2_res1(xyz2, xyz0, feat1, nbr2, fps_idx=idx2)
        nbr2_self = knn_self(xyz2, min(ns, n2))
        feat2 = self.enc2_res2(xyz2, xyz2, feat2, nbr2_self)

        # ---- Encoder Level 3 ----
        n3 = max(N // 4, 1)
        idx3 = farthest_point_sample(xyz2, n3)
        xyz3 = index_points(xyz2, idx3)
        nbr3 = knn_cross(xyz3, xyz2, min(ns, n2))
        feat3 = self.enc3_res1(xyz3, xyz2, feat2, nbr3, fps_idx=idx3)
        nbr3_self = knn_self(xyz3, min(ns, n3))
        feat3 = self.enc3_res2(xyz3, xyz3, feat3, nbr3_self)

        # ---- Encoder Level 4 (deformable) ----
        n4 = max(N // 8, 1)
        idx4 = farthest_point_sample(xyz3, n4)
        xyz4 = index_points(xyz3, idx4)
        nbr4 = knn_cross(xyz4, xyz3, min(ns, n3))
        feat4 = self.enc4_res1(xyz4, xyz3, feat3, nbr4, fps_idx=idx4)
        nbr4_self = knn_self(xyz4, min(ns, n4))
        feat4 = self.enc4_res2(xyz4, xyz4, feat4, nbr4_self)

        # ---- Encoder Level 5 (deformable) ----
        n5 = max(N // 16, 1)
        idx5 = farthest_point_sample(xyz4, n5)
        xyz5 = index_points(xyz4, idx5)
        nbr5 = knn_cross(xyz5, xyz4, min(ns, n4))
        feat5 = self.enc5_res1(xyz5, xyz4, feat4, nbr5, fps_idx=idx5)
        nbr5_self = knn_self(xyz5, min(ns, n5))
        feat5 = self.enc5_res2(xyz5, xyz5, feat5, nbr5_self)

        # ---- Decoder ----
        up4 = self._nn_upsample(xyz4, xyz5, feat5)
        dec4 = self.dec4(torch.cat([up4, feat4], dim=-1))

        up3 = self._nn_upsample(xyz3, xyz4, dec4)
        dec3 = self.dec3(torch.cat([up3, feat3], dim=-1))

        up2 = self._nn_upsample(xyz2, xyz3, dec3)
        dec2 = self.dec2(torch.cat([up2, feat2], dim=-1))

        up1 = self._nn_upsample(xyz0, xyz2, dec2)
        dec1 = self.dec1(torch.cat([up1, feat1], dim=-1))

        # ---- Head ----
        return self.head(dec1.reshape(B * N, -1)).reshape(B, N, -1)


# ---------------------------------------------------------------------------
#  Engine
# ---------------------------------------------------------------------------

@register_engine("kpconv_full")
class KPConvFullEngine:

    @staticmethod
    def build_model(num_classes, in_channels, **kwargs):
        mcfg = kwargs.get("model_cfg", {})
        return KPConvFullSemSeg(
            num_classes=num_classes,
            in_channels=in_channels,
            first_feat_dim=mcfg.get("first_feat_dim", 128),
            num_kpoints=mcfg.get("num_kpoints", 15),
            base_radius=mcfg.get("base_radius", 0.1),
            nsample=mcfg.get("nsample", 32),
            influence=mcfg.get("influence", "linear"),
            modulated=mcfg.get("modulated", False),
            deform_fitting_power=mcfg.get("deform_fitting_power", 1.0),
            repulse_extent=mcfg.get("repulse_extent", 1.2),
        )

    @staticmethod
    def build_dataset(file_paths, cfg, split, lofg, feature_columns, label_offset):
        mcfg = cfg["models"]["kpconv_full"]
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
        mcfg = cfg["models"]["kpconv_full"]
        # Separate learning rate for deformation parameters
        deform_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'offset_mlp' in name or 'modulation_mlp' in name:
                deform_params.append(param)
            else:
                other_params.append(param)

        lr = mcfg["lr"]
        deform_lr_factor = mcfg.get("deform_lr_factor", 0.1)

        return optim.Adam([
            {'params': other_params, 'lr': lr},
            {'params': deform_params, 'lr': lr * deform_lr_factor},
        ], weight_decay=mcfg["weight_decay"])

    @staticmethod
    def get_scheduler(optimizer, cfg, **_kwargs):
        mcfg = cfg["models"]["kpconv_full"]
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
        feat = feat.float().to(device)
        labels = labels.long().to(device)
        pred = model(feat.transpose(2, 1))
        num_classes = pred.shape[-1]

        # Main loss
        loss = criterion(pred.reshape(-1, num_classes), labels.reshape(-1))

        # Deformation regularization loss
        if hasattr(model, 'get_deformation_loss'):
            deform_loss = model.get_deformation_loss()
            loss = loss + 0.1 * deform_loss

        return loss, pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def eval_step(model, batch, device):
        feat, labels = batch
        feat = feat.float().to(device)
        labels = labels.long().to(device)
        with torch.no_grad():
            pred = model(feat.transpose(2, 1))
        num_classes = pred.shape[-1]
        return pred.reshape(-1, num_classes), labels.reshape(-1)

    @staticmethod
    def get_epochs(cfg):
        return cfg["models"]["kpconv_full"]["epochs"]

    @staticmethod
    def get_batch_size(cfg):
        return cfg["models"]["kpconv_full"]["batch_size"]
