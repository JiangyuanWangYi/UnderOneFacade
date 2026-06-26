"""
Unified facade point cloud datasets.
Reads .asc (Nottingham) and .npy (ZAHA, Singapore) into the same (N,8) format:
  [X, Y, Z, R, G, B, Intensity, Label]

Provides:
  - BlockDataset: random spatial blocks for PN2 / DGCNN
  - PointOffsetDataset: offset-batched point sets for PTv1
  - SparseVoxelDataset: voxelized input for PTv3
  - OctreeDataset: octree-based input for OctFormer
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.misc import remap_lofg3_to_lofg2
from utils.load_data import load_file


# ---------------------------------------------------------------------------
# Stratified Sampling (ensures minority classes are represented)
# ---------------------------------------------------------------------------

def stratified_sample(labels: np.ndarray, num_points: int,
                      min_per_class: int = 200) -> np.ndarray:
    """
    Sample points ensuring each class gets at least min_per_class points.
    Critical for imbalanced datasets like LoFG3 (15 classes).
    
    Args:
        labels: (N,) class labels
        num_points: target number of points to sample
        min_per_class: minimum points per class (default 200)
    
    Returns:
        indices: (num_points,) selected point indices
    """
    n = len(labels)
    if n <= num_points:
        return np.arange(n)
    
    unique_cls = np.unique(labels)
    # Filter out ignore label (255) if present
    unique_cls = unique_cls[unique_cls != 255]
    n_cls = len(unique_cls)
    
    if n_cls == 0:
        return np.random.choice(n, num_points, replace=False)
    
    # Calculate quota per class
    quota = min(min_per_class, num_points // max(n_cls, 1))
    quota = max(quota, 1)
    
    chosen = []
    for c in unique_cls:
        idx = np.where(labels == c)[0]
        take = min(len(idx), quota)
        if take > 0:
            chosen.append(np.random.choice(idx, take, replace=False))
    
    if len(chosen) == 0:
        return np.random.choice(n, num_points, replace=False)
    
    chosen = np.concatenate(chosen)
    
    # Fill remaining slots with random sampling
    remaining = num_points - len(chosen)
    if remaining > 0:
        mask = np.ones(n, dtype=bool)
        mask[chosen] = False
        pool = np.where(mask)[0]
        if len(pool) > 0:
            extra = np.random.choice(pool, min(remaining, len(pool)),
                                    replace=False)
            chosen = np.concatenate([chosen, extra])
    
    # If still not enough (rare), sample with replacement
    if len(chosen) < num_points:
        extra_needed = num_points - len(chosen)
        extra = np.random.choice(n, extra_needed, replace=True)
        chosen = np.concatenate([chosen, extra])
    
    return chosen[:num_points]


# ---------------------------------------------------------------------------
# File Subsampling (for limited labeled data experiments)
# ---------------------------------------------------------------------------

def subsample_files(file_paths: list, percent: float, seed: int = 999,
                    stratified: bool = False) -> list:
    """
    Subsample files to simulate limited labeled data scenarios.
    
    Args:
        file_paths: List of file paths to subsample from
        percent: Percentage of files to keep (1-100)
        seed: Random seed for reproducibility
        stratified: If True, attempt to preserve class distribution (requires loading files)
    
    Returns:
        Subsampled list of file paths
    """
    if percent >= 100.0:
        return file_paths
    
    np.random.seed(seed)
    n_total = len(file_paths)
    n_keep = max(1, int(n_total * percent / 100.0))
    
    if not stratified:
        # Simple random sampling
        indices = np.random.choice(n_total, n_keep, replace=False)
        return [file_paths[i] for i in sorted(indices)]
    
    # Stratified sampling: try to balance class distribution
    # Load a sample of labels from each file to estimate class distribution
    file_class_counts = []
    for fp in file_paths:
        try:
            arr = load_file(fp)
            labels = arr[:, 7].astype(np.int64)
            unique, counts = np.unique(labels, return_counts=True)
            file_class_counts.append({int(u): int(c) for u, c in zip(unique, counts)})
        except Exception:
            file_class_counts.append({})
    
    # Score files by their coverage of rare classes
    all_classes = set()
    for fcc in file_class_counts:
        all_classes.update(fcc.keys())
    
    # Compute global class frequencies
    global_counts = {c: 0 for c in all_classes}
    for fcc in file_class_counts:
        for c, cnt in fcc.items():
            global_counts[c] += cnt
    
    total_points = sum(global_counts.values())
    if total_points == 0:
        # Fallback to random
        indices = np.random.choice(n_total, n_keep, replace=False)
        return [file_paths[i] for i in sorted(indices)]
    
    # Inverse frequency weighting: files with rare classes get higher scores
    class_weights = {c: total_points / (cnt + 1) for c, cnt in global_counts.items()}
    
    file_scores = []
    for i, fcc in enumerate(file_class_counts):
        score = sum(class_weights.get(c, 0) * cnt for c, cnt in fcc.items())
        file_scores.append((i, score))
    
    # Sort by score (descending) and add randomness
    file_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top n_keep with some randomization in the selection
    top_candidates = file_scores[:min(n_keep * 2, n_total)]
    np.random.shuffle(top_candidates)
    selected_indices = [fs[0] for fs in top_candidates[:n_keep]]
    
    return [file_paths[i] for i in sorted(selected_indices)]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_all_files(file_paths: list) -> tuple:
    """Load multiple files, return lists of (xyz, features, labels) per scene."""
    all_data = []
    for fp in file_paths:
        arr = load_file(fp)
        all_data.append(arr)
    return all_data


# ---------------------------------------------------------------------------
# Block dataset for PN2 / DGCNN
# ---------------------------------------------------------------------------

class BlockDataset(Dataset):
    """
    Random spatial block sampler.
    Returns (feat, label) tensors per block.
      feat: (num_point, C) where C = len(feature_columns)
      label: (num_point,)
    """

    def __init__(
        self,
        file_paths: list,
        num_point: int = 4096,
        block_size: float = 1.0,
        lofg: str = "lofg3",
        feature_columns: list = None,
        label_offset: int = 1,
        sample_rate: float = 1.0,
    ):
        self.num_point = num_point
        self.block_size = block_size
        self.lofg = lofg
        self.feature_columns = feature_columns or [0, 1, 2]
        self.label_offset = label_offset

        self.room_xyz = []
        self.room_feat = []
        self.room_labels = []
        num_points_list = []

        for fp in file_paths:
            arr = load_file(fp)
            xyz = arr[:, :3]
            feat = arr[:, self.feature_columns]
            labels = arr[:, 7].astype(np.int64) - self.label_offset

            if lofg == "lofg2":
                labels = remap_lofg3_to_lofg2(np.clip(labels, 0, 14))
            else:
                labels = np.clip(labels, 0, 14)

            self.room_xyz.append(xyz)
            self.room_feat.append(feat)
            self.room_labels.append(labels)
            num_points_list.append(xyz.shape[0])

        n_arr = np.array(num_points_list, dtype=np.float64)
        sample_prob = n_arr / n_arr.sum()
        num_iter = int(n_arr.sum() * sample_rate / num_point)

        room_idxs = []
        for i in range(len(self.room_xyz)):
            room_idxs.extend([i] * int(round(sample_prob[i] * num_iter)))
        self.room_idxs = np.array(room_idxs) if room_idxs else np.array([0])

        print(f"  BlockDataset [{lofg}]: {len(file_paths)} files, "
              f"{len(self.room_idxs)} blocks/epoch, {len(self.feature_columns)}ch")

    def __len__(self):
        return len(self.room_idxs)

    def __getitem__(self, idx):
        ri = self.room_idxs[idx]
        xyz = self.room_xyz[ri]
        feat = self.room_feat[ri]
        labels = self.room_labels[ri]

        for _ in range(10):
            ctr = xyz[np.random.randint(xyz.shape[0])]
            half = self.block_size / 2
            mask = (
                (xyz[:, 0] >= ctr[0] - half) & (xyz[:, 0] <= ctr[0] + half) &
                (xyz[:, 1] >= ctr[1] - half) & (xyz[:, 1] <= ctr[1] + half)
            )
            if mask.sum() > 50:
                break

        bfeat = feat[mask]
        blbl = labels[mask]
        bxyz = xyz[mask]

        n = bfeat.shape[0]
        if n == 0:
            choice = np.random.choice(feat.shape[0], self.num_point,
                                      replace=feat.shape[0] < self.num_point)
            bfeat = feat[choice]
            blbl = labels[choice]
            bxyz = xyz[choice]
        else:
            choice = np.random.choice(n, self.num_point, replace=(n < self.num_point))
            bfeat = bfeat[choice]
            blbl = blbl[choice]
            bxyz = bxyz[choice]

        # Normalize xyz columns within block
        bxyz_norm = bxyz - bxyz.mean(0)
        d = np.sqrt((bxyz_norm ** 2).sum(1)).max()
        bxyz_norm = bxyz_norm / (d + 1e-6)

        # For xyz-only features, replace with normalized xyz
        if self.feature_columns == [0, 1, 2]:
            bfeat = bxyz_norm.astype(np.float32)
        elif self.feature_columns == [0, 1, 2, 0, 1, 2]:
            # Duplicated xyz case - repeat normalized xyz
            bfeat = np.concatenate([bxyz_norm, bxyz_norm], axis=1).astype(np.float32)
        else:
            # Replace xyz part with normalized, keep other features
            bfeat = bfeat.copy()
            bfeat[:, :3] = bxyz_norm
            # Normalize RGB (columns 3-5) and Intensity (column 6) to [0,1]
            # Data is known to be in 0-255 range for all datasets
            if len(self.feature_columns) > 3:
                rgbi_idx = [i for i, c in enumerate(self.feature_columns) if 3 <= c <= 6]
                for ri_idx in rgbi_idx:
                    bfeat[:, ri_idx] /= 255.0
            bfeat = bfeat.astype(np.float32)

        return (
            torch.from_numpy(np.ascontiguousarray(bfeat)),
            torch.from_numpy(np.ascontiguousarray(blbl)),
        )


# ---------------------------------------------------------------------------
# Point offset dataset for PTv1
# ---------------------------------------------------------------------------

class PointOffsetDataset(Dataset):
    """
    Returns full scenes (subsampled to num_point) with offset-based batching for PTv1.
    Each item: (coords, feat, labels) as numpy.
    Use point_collate_fn for DataLoader.
    """

    def __init__(
        self,
        file_paths: list,
        num_point: int = 8192,
        lofg: str = "lofg3",
        feature_columns: list = None,
        label_offset: int = 1,
        repeat: int = 1,
    ):
        self.num_point = num_point
        self.lofg = lofg
        self.feature_columns = feature_columns or [0, 1, 2]
        self.label_offset = label_offset
        self.repeat = repeat

        self.scenes = []
        for fp in file_paths:
            arr = load_file(fp)
            self.scenes.append(arr)

        print(f"  PointOffsetDataset [{lofg}]: {len(file_paths)} files × {repeat} repeats "
              f"= {len(file_paths) * repeat} samples/epoch, {num_point} pts/sample")

    def __len__(self):
        return len(self.scenes) * self.repeat

    def __getitem__(self, idx):
        arr = self.scenes[idx % len(self.scenes)]
        n = arr.shape[0]
        
        # Compute labels first for stratified sampling
        labels_raw = arr[:, 7].astype(np.int64) - self.label_offset
        if self.lofg == "lofg2":
            labels_for_sampling = remap_lofg3_to_lofg2(np.clip(labels_raw, 0, 14))
        else:
            labels_for_sampling = np.clip(labels_raw, 0, 14)
        
        # Use stratified sampling to ensure minority classes are represented
        if n > self.num_point:
            choice = stratified_sample(labels_for_sampling, self.num_point,
                                       min_per_class=200)
        else:
            choice = np.random.choice(n, self.num_point, replace=True)
        arr = arr[choice]

        coords = arr[:, :3].astype(np.float32)
        feat = arr[:, self.feature_columns].astype(np.float32)
        labels = arr[:, 7].astype(np.int64) - self.label_offset

        if self.lofg == "lofg2":
            labels = remap_lofg3_to_lofg2(np.clip(labels, 0, 14))
        else:
            labels = np.clip(labels, 0, 14)

        # Normalize coords per scene
        coords -= coords.mean(0)
        d = np.sqrt((coords ** 2).sum(1)).max()
        coords /= (d + 1e-6)

        # For xyz features, feat is the normalized coords
        if self.feature_columns == [0, 1, 2]:
            feat = coords.copy()
        else:
            # Replace xyz part with normalized coords
            feat[:, :3] = coords
            # Normalize RGB (columns 3-5) and Intensity (column 6) to [0,1]
            for i, c in enumerate(self.feature_columns):
                if 3 <= c <= 6:
                    feat[:, i] /= 255.0

        return coords, feat, labels


def point_collate_fn(batch):
    """Collate for offset-based batching (PTv1 style)."""
    coords_list, feat_list, label_list = [], [], []
    offset = []
    count = 0
    for coords, feat, labels in batch:
        coords_list.append(torch.from_numpy(coords))
        feat_list.append(torch.from_numpy(feat))
        label_list.append(torch.from_numpy(labels))
        count += coords.shape[0]
        offset.append(count)

    return (
        torch.cat(coords_list, dim=0).float(),
        torch.cat(feat_list, dim=0).float(),
        torch.cat(label_list, dim=0).long(),
        torch.tensor(offset, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Sparse voxel dataset for PTv3
# ---------------------------------------------------------------------------

class SparseVoxelDataset(Dataset):
    """
    Returns voxelized point cloud dicts for PTv3.
    Each item: dict with 'coord', 'feat', 'label', 'grid_size'.
    """

    def __init__(
        self,
        file_paths: list,
        voxel_size: float = 0.05,
        num_point: int = 80000,
        lofg: str = "lofg3",
        feature_columns: list = None,
        label_offset: int = 1,
    ):
        self.voxel_size = voxel_size
        self.num_point = num_point
        self.lofg = lofg
        self.feature_columns = feature_columns or [0, 1, 2]
        self.label_offset = label_offset

        self.scenes = []
        for fp in file_paths:
            arr = load_file(fp)
            self.scenes.append(arr)

        print(f"  SparseVoxelDataset [{lofg}]: {len(file_paths)} files, "
              f"voxel={voxel_size}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        arr = self.scenes[idx]
        n = arr.shape[0]
        
        # Compute labels first for stratified sampling
        labels_raw = arr[:, 7].astype(np.int64) - self.label_offset
        if self.lofg == "lofg2":
            labels_for_sampling = remap_lofg3_to_lofg2(np.clip(labels_raw, 0, 14))
        else:
            labels_for_sampling = np.clip(labels_raw, 0, 14)
        
        # Use stratified sampling to ensure minority classes are represented
        if n > self.num_point:
            choice = stratified_sample(labels_for_sampling, self.num_point,
                                       min_per_class=200)
            arr = arr[choice]

        coord = arr[:, :3].astype(np.float32)
        feat = arr[:, self.feature_columns].astype(np.float32)
        labels = arr[:, 7].astype(np.int64) - self.label_offset

        if self.lofg == "lofg2":
            labels = remap_lofg3_to_lofg2(np.clip(labels, 0, 14))
        else:
            labels = np.clip(labels, 0, 14)

        # Normalize coordinates: center XY, shift Z to start at 0
        coord[:, :2] -= coord[:, :2].mean(0)
        coord[:, 2] -= coord[:, 2].min()

        # For xyz-only features, use the normalized coords
        if self.feature_columns == [0, 1, 2]:
            feat = coord.copy()
        else:
            # Replace xyz part with normalized coords
            feat[:, :3] = coord
            # Normalize RGB (columns 3-5) and Intensity (column 6) to [0,1]
            for i, c in enumerate(self.feature_columns):
                if 3 <= c <= 6:
                    feat[:, i] /= 255.0

        # Compute grid coordinates from normalized coords
        grid_coord = np.floor(coord / self.voxel_size).astype(np.int64)
        grid_coord -= grid_coord.min(0)

        return {
            "coord": torch.from_numpy(coord).float(),
            "grid_coord": torch.from_numpy(grid_coord).long(),
            "feat": torch.from_numpy(feat).float(),
            "label": torch.from_numpy(labels).long(),
        }


def sparse_collate_fn(batch):
    """Collate for sparse voxel batching (PTv3 style)."""
    coords, grid_coords, feats, labels = [], [], [], []
    offset = []
    count = 0
    for item in batch:
        n = item["coord"].shape[0]
        coords.append(item["coord"])
        grid_coords.append(item["grid_coord"])
        feats.append(item["feat"])
        labels.append(item["label"])
        count += n
        offset.append(count)

    return {
        "coord": torch.cat(coords, dim=0),
        "grid_coord": torch.cat(grid_coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "label": torch.cat(labels, dim=0),
        "offset": torch.tensor(offset, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Octree dataset for OctFormer
# ---------------------------------------------------------------------------

class OctreeDataset(Dataset):
    """
    Returns point cloud data for OctFormer octree construction.
    Each item: dict with 'coord', 'feat', 'label'.
    Octree is built in the engine's collate/train_step.
    """

    def __init__(
        self,
        file_paths: list,
        num_point: int = 80000,
        lofg: str = "lofg3",
        feature_columns: list = None,
        label_offset: int = 1,
    ):
        self.num_point = num_point
        self.lofg = lofg
        self.feature_columns = feature_columns or [0, 1, 2]
        self.label_offset = label_offset

        self.scenes = []
        for fp in file_paths:
            arr = load_file(fp)
            self.scenes.append(arr)

        print(f"  OctreeDataset [{lofg}]: {len(file_paths)} files, "
              f"{num_point} pts/sample")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        arr = self.scenes[idx]
        n = arr.shape[0]
        
        # Compute labels first for stratified sampling
        labels_raw = arr[:, 7].astype(np.int64) - self.label_offset
        if self.lofg == "lofg2":
            labels_for_sampling = remap_lofg3_to_lofg2(np.clip(labels_raw, 0, 14))
        else:
            labels_for_sampling = np.clip(labels_raw, 0, 14)
        
        # Use stratified sampling to ensure minority classes are represented
        if n > self.num_point:
            choice = stratified_sample(labels_for_sampling, self.num_point,
                                       min_per_class=200)
            arr = arr[choice]

        coord = arr[:, :3].astype(np.float32)
        feat = arr[:, self.feature_columns].astype(np.float32)
        labels = arr[:, 7].astype(np.int64) - self.label_offset

        if self.lofg == "lofg2":
            labels = remap_lofg3_to_lofg2(np.clip(labels, 0, 14))
        else:
            labels = np.clip(labels, 0, 14)

        # Normalize to [-1, 1] for octree (matches OCNN / JAB convention)
        center = (coord.min(0) + coord.max(0)) / 2
        scale = (coord.max(0) - coord.min(0)).max() / 2
        coord = (coord - center) / (scale + 1e-6)

        # For xyz-only features, use the normalized coords
        if self.feature_columns == [0, 1, 2]:
            feat = coord.copy()
        else:
            # Replace xyz part with normalized coords
            feat[:, :3] = coord
            # Normalize RGB (columns 3-5) and Intensity (column 6) to [0,1]
            for i, c in enumerate(self.feature_columns):
                if 3 <= c <= 6:
                    feat[:, i] /= 255.0

        return {
            "coord": torch.from_numpy(coord).float(),
            "feat": torch.from_numpy(feat).float(),
            "label": torch.from_numpy(labels).long(),
        }


def octree_collate_fn(batch):
    """Simple collate for OctFormer -- stacks dicts."""
    coords, feats, labels = [], [], []
    offset = []
    count = 0
    for item in batch:
        n = item["coord"].shape[0]
        coords.append(item["coord"])
        feats.append(item["feat"])
        labels.append(item["label"])
        count += n
        offset.append(count)

    return {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "label": torch.cat(labels, dim=0),
        "offset": torch.tensor(offset, dtype=torch.long),
    }
