import os
import numpy as np
from PIL import Image


def load_target_frames(targets_dir: str, num_frames: int) -> np.ndarray:
    """Load preprocessed target frames as a numpy array.

    Returns array of shape [num_frames, 256, 256] with values in [0, 1].
    """
    frames = []
    for i in range(1, num_frames + 1):
        path = os.path.join(targets_dir, f"frame_{i:05d}.png")
        if not os.path.exists(path):
            break
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(arr)
    return np.stack(frames)


def normalize_rows_for_causal(target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize target frame rows to sum to 1, respecting the causal mask.

    Input: [H, W] array in [0, 1]
    Output: [H, W] array where each row i has values only in columns [0..i],
            and that slice sums to 1.
    """
    h, w = target.shape
    out = np.zeros_like(target)
    for i in range(h):
        row = target[i, : i + 1].copy()
        row = row + eps
        row = row / row.sum()
        out[i, : i + 1] = row
    return out


def preprocess_targets(frames: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Preprocess all target frames: apply causal mask and row-normalize.

    Input: [N, 256, 256]
    Output: [N, 256, 256] row-normalized with causal mask
    """
    return np.stack([normalize_rows_for_causal(f, eps) for f in frames])
