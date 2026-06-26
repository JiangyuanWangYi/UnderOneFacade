import numpy as np

POINT_CLOUD_COLUMNS = ("X", "Y", "Z", "R", "G", "B", "Intensity", "Label")
NUM_POINT_CLOUD_COLUMNS = 8


def read_point_cloud(path: str) -> np.ndarray:
    """Read a facade point cloud file into (N, 8) float32: XYZRGBI + label."""
    if path.endswith(".npy"):
        data = np.load(path).astype(np.float32)
    elif path.endswith(".asc"):
        data = np.loadtxt(path, skiprows=1, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")

    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[1] < NUM_POINT_CLOUD_COLUMNS:
        pad = np.zeros(
            (data.shape[0], NUM_POINT_CLOUD_COLUMNS - data.shape[1]),
            dtype=np.float32,
        )
        data = np.concatenate([data, pad], axis=1)
    return data[:, :NUM_POINT_CLOUD_COLUMNS]


def load_file(path: str) -> np.ndarray:
    """Alias for read_point_cloud (backward compatible with datasets)."""
    return read_point_cloud(path)
