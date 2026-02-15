import argparse
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from transformers import GPT2Model

from utils import load_target_frames, preprocess_targets

FRAMES_DIR = "frames"
TARGETS_DIR = os.path.join(FRAMES_DIR, "targets")
OUTPUT_DIR = os.path.join(FRAMES_DIR, "output")

NUM_HEADS = 25
SEQ_LEN = 256
HIDDEN_DIM = 1600
D_HEAD = HIDDEN_DIM // NUM_HEADS  # 64


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular boolean mask [seq_len, seq_len]."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def targets_to_desired_logits(targets_np, device):
    """Convert row-normalized target frames to desired pre-softmax logits.

    For each target (a valid probability distribution), the desired logits
    are log(target), with row-wise centering to resolve the softmax's
    shift invariance.
    """
    desired = np.log(targets_np + 1e-8)
    for i in range(desired.shape[0]):
        for r in range(SEQ_LEN):
            row = desired[i, r, : r + 1]
            desired[i, r, : r + 1] = row - row.mean()
    return torch.tensor(desired, dtype=torch.float32, device=device)


def normalize_attention_for_display(attn_map: np.ndarray, blur_sigma: float = 1.5) -> np.ndarray:
    """Convert a causal attention map to a displayable image.

    Per-row z-score normalization reveals relative attention patterns,
    then Gaussian blur smooths out horizontal streak artifacts.
    Upper triangle set to 0 (black in magma colormap).
    """
    h, w = attn_map.shape
    display = np.zeros((h, w))

    for i in range(1, h):
        row = attn_map[i, : i + 1]
        mu = row.mean()
        std = row.std()
        if std > 1e-10:
            display[i, : i + 1] = (row - mu) / std

    mask = np.tril(np.ones((h, w), dtype=bool))
    causal_vals = display[mask]
    vmin = np.percentile(causal_vals, 1)
    vmax = np.percentile(causal_vals, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-8

    display = np.clip((display - vmin) / (vmax - vmin), 0, 1)

    if blur_sigma > 0:
        display = gaussian_filter(display, sigma=blur_sigma)

    display = np.where(mask, display, 0.0)
    return display


def optimize_single_head(
    model,
    all_targets: np.ndarray,
    device: torch.device,
    head_idx: int = 0,
    layer_idx: int = 0,
    lr: float = 0.1,
    lr_min: float = 1e-4,
    steps: int = 1000,
    init_scale: float = 0.1,
    chunk_size: int = 64,
    metrics: dict | None = None,
):
    """Optimize each frame independently against a single attention head.

    Batches multiple frames together for GPU efficiency, but each frame
    gets its own input embedding (no overconstrained sharing).
    Uses extracted head-specific Q/K weights for ~25x faster forward pass.
    """
    block = model.h[layer_idx].float()

    # Extract head-specific Q and K projection weights
    q_start = head_idx * D_HEAD
    k_start = HIDDEN_DIM + head_idx * D_HEAD
    w_q = block.attn.c_attn.weight[:, q_start : q_start + D_HEAD].detach()
    b_q = block.attn.c_attn.bias[q_start : q_start + D_HEAD].detach()
    w_k = block.attn.c_attn.weight[:, k_start : k_start + D_HEAD].detach()
    b_k = block.attn.c_attn.bias[k_start : k_start + D_HEAD].detach()
    ln_weight = block.ln_1.weight.detach()
    ln_bias = block.ln_1.bias.detach()
    ln_eps = block.ln_1.eps

    pos_embeds = model.wpe(torch.arange(SEQ_LEN, device=device).unsqueeze(0)).detach().float()
    causal_float = build_causal_mask(SEQ_LEN, device).float()
    row_counts = causal_float.sum(dim=-1)
    causal_bias = block.attn.bias[0, 0, :SEQ_LEN, :SEQ_LEN]

    targets_np = preprocess_targets(all_targets)
    desired_all = targets_to_desired_logits(targets_np, device)

    num_frames = len(all_targets)
    num_chunks = math.ceil(num_frames / chunk_size)
    cmap = plt.get_cmap("magma")
    from PIL import Image

    num_seeds = 3
    explore_steps = steps // 4
    refine_steps = steps - explore_steps

    print(f"Single-head mode: head {head_idx}, layer {layer_idx}")
    print(f"Processing {num_frames} frames in {num_chunks} chunks of {chunk_size}")
    print(f"Strategy: {num_seeds} seeds x {explore_steps} explore + {refine_steps} refine = {steps} effective steps")

    run_start = time.time()

    for chunk_idx in tqdm(range(num_chunks), desc="Optimizing"):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_frames)
        B = end - start

        desired = desired_all[start:end]

        def run_optimization(init_data, num_steps, lr_used):
            """Run optimization and return (final_input, per_frame_losses)."""
            inp = torch.nn.Parameter(init_data.clone())
            opt = torch.optim.Adam([inp], lr=lr_used)
            sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(num_steps // 4, 50), T_mult=1, eta_min=lr_min
            )
            for step in range(num_steps):
                opt.zero_grad()
                h_norm = F.layer_norm(inp + pos_embeds, (HIDDEN_DIM,), ln_weight, ln_bias, ln_eps)
                q = h_norm @ w_q + b_q
                k = h_norm @ w_k + b_k
                lo = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(D_HEAD)
                m = lo * causal_float
                rm = m.sum(dim=-1) / row_counts
                c = (lo - rm.unsqueeze(-1)) * causal_float
                mse = ((c - desired) ** 2 * causal_float).sum(dim=(-1, -2)) / causal_float.sum()
                mse.mean().backward()
                opt.step()
                sched.step()
            return inp.data, mse.detach()

        # Phase 1: Explore multiple seeds
        best_input = None
        best_loss = float("inf")
        for seed in range(num_seeds):
            init = torch.randn(B, SEQ_LEN, HIDDEN_DIM, device=device) * init_scale
            optimized, mse = run_optimization(init, explore_steps, lr)
            avg = mse.mean().item()
            if avg < best_loss:
                best_loss = avg
                best_input = optimized

        # Phase 2: Refine from best seed for remaining steps
        inp_data, mse_per_frame = run_optimization(best_input, refine_steps, lr)

        # Extract final attention maps and save frames
        with torch.no_grad():
            hidden = inp_data + pos_embeds
            h_norm = F.layer_norm(hidden, (HIDDEN_DIM,), ln_weight, ln_bias, ln_eps)
            q = h_norm @ w_q + b_q
            k = h_norm @ w_k + b_k
            logits = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(D_HEAD)
            mask_value = torch.finfo(logits.dtype).min
            masked_logits = torch.where(causal_bias, logits, mask_value)
            attn_maps = F.softmax(masked_logits, dim=-1).cpu().numpy()

        for i in range(B):
            frame_idx = start + i + 1  # 1-indexed
            display = normalize_attention_for_display(attn_maps[i])
            colored = (cmap(display)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(colored).save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.png"))

        if metrics is not None:
            metrics["chunks"].append(
                {
                    "chunk_idx": chunk_idx,
                    "frame_range": [start, end],
                    "final_loss": mse_per_frame.mean().item(),
                    "per_frame_losses": mse_per_frame.cpu().tolist(),
                    "time": time.time() - run_start,
                }
            )

        if chunk_idx % 5 == 0:
            avg_loss = mse_per_frame.mean().item()
            elapsed = time.time() - run_start
            eta = elapsed / (chunk_idx + 1) * (num_chunks - chunk_idx - 1)
            tqdm.write(
                f"  Chunk {chunk_idx}/{num_chunks} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )

    block.half()
    total_time = time.time() - run_start
    print(f"Done! {num_frames} frames in {total_time:.0f}s ({total_time / num_frames:.2f}s/frame)")
    return total_time


def main():
    parser = argparse.ArgumentParser(description="Optimize GPT-2 XL attention maps for Bad Apple")
    parser.add_argument("--steps", type=int, default=1500, help="Optimization steps per frame")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-4, help="Minimum learning rate")
    parser.add_argument("--init-scale", type=float, default=0.1, help="Input embedding init scale")
    parser.add_argument("--head", type=int, default=0, help="Which attention head to target")
    parser.add_argument("--chunk-size", type=int, default=64, help="Frames per GPU batch")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_frames_files = sorted(f for f in os.listdir(TARGETS_DIR) if f.endswith(".png"))
    total_frames = len(all_frames_files)
    print(f"Found {total_frames} target frames")

    print("Loading GPT-2 XL...")
    model = GPT2Model.from_pretrained("gpt2-xl", torch_dtype=torch.float16, attn_implementation="eager")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"Model loaded on {device}")

    print("Loading target frames...")
    all_targets = load_target_frames(TARGETS_DIR, total_frames)
    print(f"Loaded {len(all_targets)} target frames, shape: {all_targets.shape}")

    metrics = {
        "args": vars(args),
        "total_frames": total_frames,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": [],
    }

    total_time = optimize_single_head(
        model=model,
        all_targets=all_targets,
        device=device,
        head_idx=args.head,
        layer_idx=0,
        lr=args.lr,
        lr_min=args.lr_min,
        steps=args.steps,
        init_scale=args.init_scale,
        chunk_size=args.chunk_size,
        metrics=metrics,
    )

    metrics["total_time_seconds"] = total_time
    metrics["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
