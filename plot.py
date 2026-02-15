import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PLOT_DIR = "plots"
OUTPUT_DIR = os.path.join("frames", "output")
TARGETS_DIR = os.path.join("frames", "targets")

# Monokai Classic Pro
BG = "#272822"
FG = "#F8F8F2"
COMMENT = "#75715E"
GUTTER = "#3E3D32"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GUTTER,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": COMMENT,
        "ytick.color": COMMENT,
        "grid.color": GUTTER,
        "grid.alpha": 0.6,
        "font.family": "monospace",
        "font.size": 11,
    }
)

PINK = "#F92672"  # monokai pink/magenta (primary)
GREEN = "#A6E22E"  # monokai green (secondary)
YELLOW = "#E6DB74"  # monokai yellow
PURPLE = "#AE81FF"  # monokai purple


def load_metrics():
    with open(os.path.join(OUTPUT_DIR, "metrics.json")) as f:
        return json.load(f)


def get_all_frame_losses(metrics):
    """Extract per-frame losses in frame order."""
    losses = []
    for chunk in metrics["chunks"]:
        losses.extend(chunk["per_frame_losses"])
    return np.array(losses)


def load_frame(directory, frame_idx):
    """Load a frame image (1-indexed)."""
    path = os.path.join(directory, f"frame_{frame_idx:05d}.png")
    return np.array(Image.open(path))


# ── Plot 1: Per-frame loss across the entire video ─────────────────────


def plot_loss_timeline(losses):
    fig, ax = plt.subplots(figsize=(14, 4))

    frames = np.arange(1, len(losses) + 1)
    ax.fill_between(frames, losses, alpha=0.3, color=PINK)
    ax.plot(frames, losses, linewidth=0.4, color=PINK, alpha=0.8)

    # Rolling average
    window = 50
    rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(
        frames[window // 2 : window // 2 + len(rolling)],
        rolling,
        linewidth=2,
        color=GREEN,
        label=f"{window}-frame rolling avg",
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("MSE Loss (logit-space)")
    ax.set_xlim(1, len(losses))
    ax.set_ylim(0, min(losses.max() * 1.1, 2.5))
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "loss_timeline.png"), dpi=200)
    plt.close()
    print("Saved loss_timeline.png")


# ── Plot 2: Loss distribution histogram ────────────────────────────────


def plot_loss_histogram(losses):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Clip for display (a few outliers above 2)
    clipped = np.clip(losses, 0, 2.0)
    counts, bins, patches = ax.hist(clipped, bins=80, color=PINK, alpha=0.85, edgecolor="none")

    # Color the tail differently
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 1.0:
            patch.set_facecolor(PINK)
            patch.set_alpha(0.85)

    median = np.median(losses)
    mean = np.mean(losses)
    ax.axvline(median, color=GREEN, linewidth=2, linestyle="--", label=f"Median: {median:.3f}")
    ax.axvline(mean, color=GREEN, linewidth=2, linestyle=":", label=f"Mean: {mean:.3f}")

    ax.set_xlabel("MSE Loss (logit-space)")
    ax.set_ylabel("Frame count")
    ax.legend(loc="upper right", framealpha=0.3)
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "loss_histogram.png"), dpi=200)
    plt.close()
    print("Saved loss_histogram.png")


# ── Plot 3: Target vs Output comparison grid ───────────────────────────


def plot_comparison_grid(losses):
    # Pick frames that tell a story: easy → medium → hard
    showcase = [
        (510, "Simple silhouette"),
        (671, "Detailed figure"),
        (1181, "Complex scene"),
        (919, "Sharp transition"),
        (1825, "Failure case"),
        (2732, "Hardest frame"),
    ]

    fig, axes = plt.subplots(2, len(showcase), figsize=(3.2 * len(showcase), 6.8))

    for col, (frame_idx, label) in enumerate(showcase):
        target = load_frame(TARGETS_DIR, frame_idx)
        output = load_frame(OUTPUT_DIR, frame_idx)
        loss_val = losses[frame_idx - 1]

        axes[0, col].imshow(target, cmap="gray")
        axes[0, col].set_title(f"Target #{frame_idx}", fontsize=9, pad=4)
        axes[0, col].axis("off")

        axes[1, col].imshow(output)
        axes[1, col].set_title(f"Output (loss={loss_val:.3f})", fontsize=9, pad=4)
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.08,
        0.5,
        "Target\n(grayscale)",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        va="center",
        ha="right",
        rotation=90,
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "GPT-2 XL\nAttention",
        transform=axes[1, 0].transAxes,
        fontsize=10,
        va="center",
        ha="right",
        rotation=90,
    )

    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(PLOT_DIR, "comparison_grid.png"), dpi=200)
    plt.close()
    print("Saved comparison_grid.png")


# ── Plot 4: Cumulative quality curve ───────────────────────────────────


def plot_cumulative(losses):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sorted_losses = np.sort(losses)
    percentiles = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses) * 100

    ax.plot(sorted_losses, percentiles, linewidth=2.5, color=PINK)
    ax.fill_between(sorted_losses, percentiles, alpha=0.15, color=PINK)

    # Mark key thresholds
    for threshold, label_text in [(0.1, "< 0.1"), (0.3, "< 0.3"), (0.5, "< 0.5"), (1.0, "< 1.0")]:
        pct = (losses < threshold).sum() / len(losses) * 100
        ax.plot(threshold, pct, "o", color=GREEN, markersize=8, zorder=5)
        ax.annotate(
            f"{pct:.0f}%",
            (threshold, pct),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=10,
            color=GREEN,
        )

    ax.set_xlabel("MSE Loss (logit-space)")
    ax.set_ylabel("% of frames below threshold")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 100)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "cumulative_quality.png"), dpi=200)
    plt.close()
    print("Saved cumulative_quality.png")


# ── Plot 5: Best vs worst frames side by side ──────────────────────────


def plot_best_worst(losses):
    # Best non-trivial frames (loss > 0.001 to skip blanks)
    non_trivial = np.where(losses > 0.001)[0]
    best_indices = non_trivial[np.argsort(losses[non_trivial])[:4]]
    worst_indices = np.argsort(losses)[-4:][::-1]

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

    for col, idx in enumerate(best_indices):
        frame_idx = idx + 1
        output = load_frame(OUTPUT_DIR, frame_idx)
        axes[0, col].imshow(output)
        axes[0, col].set_title(f"#{frame_idx}\nloss={losses[idx]:.4f}", fontsize=9)
        axes[0, col].axis("off")

    for col, idx in enumerate(worst_indices):
        frame_idx = idx + 1
        output = load_frame(OUTPUT_DIR, frame_idx)
        axes[1, col].imshow(output)
        axes[1, col].set_title(f"#{frame_idx}\nloss={losses[idx]:.3f}", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.08,
        0.5,
        "Best",
        transform=axes[0, 0].transAxes,
        fontsize=13,
        fontweight="bold",
        va="center",
        ha="right",
        rotation=90,
        color=GREEN,
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "Worst",
        transform=axes[1, 0].transAxes,
        fontsize=13,
        fontweight="bold",
        va="center",
        ha="right",
        rotation=90,
        color=PINK,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "best_worst.png"), dpi=200)
    plt.close()
    print("Saved best_worst.png")


# ── Plot 6: Hero image — single frame target→attention ──────────────────


def plot_hero(losses):
    # Iconic half-split silhouette — instantly recognizable Bad Apple
    frame_idx = 1200
    target = load_frame(TARGETS_DIR, frame_idx)
    output = load_frame(OUTPUT_DIR, frame_idx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Bad Apple Frame", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(output)
    axes[1].set_title("GPT-2 XL Attention Map", fontsize=13)
    axes[1].axis("off")

    # Arrow between
    fig.text(0.5, 0.5, "\u2192", fontsize=40, ha="center", va="center", transform=fig.transFigure, color=PINK)

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(PLOT_DIR, "hero.png"), dpi=200)
    plt.close()
    print("Saved hero.png")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    metrics = load_metrics()
    losses = get_all_frame_losses(metrics)
    print(f"Loaded {len(losses)} frame losses")
    print(
        f"  Mean: {losses.mean():.4f} | Median: {np.median(losses):.4f} | "
        f"Min: {losses.min():.6f} | Max: {losses.max():.4f}"
    )

    plot_loss_timeline(losses)
    plot_loss_histogram(losses)
    plot_comparison_grid(losses)
    plot_cumulative(losses)
    plot_best_worst(losses)
    plot_hero(losses)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
