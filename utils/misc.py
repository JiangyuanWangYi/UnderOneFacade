import os
import random
import yaml
import numpy as np
import torch


LOFG3_TO_LOFG2 = np.array([0, 1, 1, 0, 2, 2, 0, 0, 0, 3, 3, 4, 1, 4, 4], dtype=np.int64)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def remap_lofg3_to_lofg2(labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(labels, 0, 14)
    return LOFG3_TO_LOFG2[clipped]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_weight_path(
    weights_root: str,
    source_countries: str,
    model_name: str,
    lofg: str,
    features: str,
) -> str:
    """Resolve weight path by convention: {root}/{source}/{model}_{lofg}_{features}.pth"""
    source_key = "_".join(sorted(source_countries.split(",")))
    filename = f"{model_name}_{lofg}_{features}.pth"
    return os.path.join(weights_root, source_key, filename)


def get_label_info(cfg: dict, lofg: str) -> dict:
    """Get num_classes and class_names from config for a given lofg level."""
    label_cfg = cfg["labels"][lofg]
    return {
        "num_classes": label_cfg["num_classes"],
        "class_names": label_cfg["names"],
    }


def get_feature_info(cfg: dict, features: str) -> dict:
    """Get feature columns and num_channels from config."""
    feat_cfg = cfg["features"][features]
    return {
        "columns": feat_cfg["columns"],
        "num_channels": feat_cfg["num_channels"],
    }


def extract_state_dict(ckpt: dict) -> dict:
    """Extract model state dict from checkpoint, handling various formats."""
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    # Strip DataParallel 'module.' prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def infer_in_channels_from_checkpoint(state_dict: dict, model_name: str) -> int:
    """Infer in_channels from checkpoint weight shapes."""
    if model_name == "pointnet2":
        key = "sa1.mlp_convs.0.weight"
        if key in state_dict:
            return state_dict[key].shape[1]
    elif model_name == "dgcnn":
        key = "conv1.0.weight"
        if key in state_dict:
            return state_dict[key].shape[1] // 2
    return None


def get_data_paths(cfg: dict, countries: str, split: str) -> list:
    """Get list of data file directories for given countries and split."""
    dirs = []
    for country in countries.split(","):
        country = country.strip()
        c_cfg = cfg["data"]["countries"][country]
        base = os.path.join(cfg["data"]["root"], c_cfg["path"])
        split_dir = os.path.join(base, c_cfg["splits"][split])
        dirs.append({"path": split_dir, "format": c_cfg["format"]})
    return dirs


def discover_files(data_dirs: list) -> list:
    """Discover all point cloud files (.npy and .asc) from data dir specs."""
    files = []
    for d in data_dirs:
        path = d["path"]
        fmt = d["format"]
        ext = f".{fmt}"
        if not os.path.isdir(path):
            print(f"Warning: directory not found: {path}")
            continue
        for f in sorted(os.listdir(path)):
            if f.endswith(ext):
                files.append(os.path.join(path, f))
    return files
