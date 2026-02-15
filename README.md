# Bad Apple but it's GPT-2 XL Attention Maps
This is the source code for the Bad Apple but it's GPT-2 XL Attention Maps project.
This silly project aims at using the GPT-2 XL attention maps as "displays" to play the [Bad Apple MV](https://www.youtube.com/watch?v=FtutLA63Cp8) on, keeping alive the decade-long tradition of playing this clip on the most random displays possible.

[![Bad Apple but it's GPT-2 XL Attention Maps](https://img.youtube.com/vi/UU14rQO6VzU/maxresdefault.jpg)](https://www.youtube.com/watch?v=UU14rQO6VzU)

[Blog post with full technical writeup](https://brayevalerien.com/blog/bad-apple-but-its-gpt2/)

# Setup and run this project
> [!NOTE]
> The following prerequisites are needed before installing this.
> - [git](https://git-scm.com/)
> - [python](https://www.python.org/)
> - [uv](https://docs.astral.sh/uv/)
> - [ffmpeg](https://ffmpeg.org/)
> - NVIDIA GPU with CUDA support

## Installation
Start by cloning this repo and `cd` into it.
```sh
git clone https://github.com/brayevalerien/bad-apple-but-its-gpt2
cd bad-apple-but-its-gpt2
```

Install the dependencies using `uv`.
```sh
uv sync
```

### Usage
There are four scripts to run in order (using `uv run python script.py`):
- [`download.py`](./download.py): uses ffmpeg to download the [Bad Apple MV](https://www.youtube.com/watch?v=FtutLA63Cp8), extract the audio, the frames and create a `./frames/targets` directory containing the 256x256 grayscale target images.
- [`optimize.py`](./optimize.py): that's the main script, it runs the training process and saves the results in `./frames/output`. Should take around 15 minutes on a decent GPU, peaking at 4.5GB of VRAM usage.
- [`assemble.py`](./assemble.py): a small helper that uses ffmpeg to assemble the generated frames and the audio into the final video.
- [`plot.py`](./plot.py): produces the plots for the blog post.

Run them in order or else they will fail.

### Settings
There are several parameters you can play around with in the `optimize.py` script. Usage:
```
usage: optimize.py [-h] [--steps STEPS] [--lr LR] [--lr-min LR_MIN]
                   [--init-scale INIT_SCALE] [--head HEAD]
                   [--chunk-size CHUNK_SIZE]

options:
  --steps STEPS           Optimization steps per frame (default: 1500)
  --lr LR                 Learning rate (default: 0.1)
  --lr-min LR_MIN         Minimum learning rate (default: 1e-4)
  --init-scale INIT_SCALE Input embedding init scale (default: 0.1)
  --head HEAD             Which attention head to target (default: 0)
  --chunk-size CHUNK_SIZE Frames per GPU batch (default: 64)
```

## How it works
Instead of teaching a model to predict the actual frame, we optimize the input embeddings such that a single attention head's 256x256 weight matrix reproduces each frame of Bad Apple.

Each frame gets its own 256x1600 input embedding, optimized against head 0 of layer 0 in GPT-2 XL (frozen).

If applied naively, this strategy alone wouldn't work great so we also apply the following improvements:
- prepare the targets: we do not optimize against the raw B&W frames, we instead row-normalize within the causal mask loss to get valid attention distribution target.
- multi-seed sampling: instead of optimizing a single random embedding, we start with 3 random embeddings, run the optimization partially, and only keep the best one to run all the steps.
- instead of running the full forward pass we extract the Q/K slices of the first head of the first layer and only run: PE > LayerNorm > matmul by Q and K > Q@K^T. We don't run the forward process past the first head either.
- use logit-space loss instead of cross-entropy loss on softmax output. This is important because the softmax will squeeze everything and cause vanishing gradients.
- post-process the result (that is a 256x256 matrix with all values very close to 0): per-row z-score normalisation > gaussian blur > percentile clipping. The result is plotted using the magma colormap... cause it looks cool.

Note that this technique could be used with any grayscale video (with enough contrast).

With these, we render 3286 frames in about 12 minutes on a RTX 5070 Ti (with the default settings).

---
Note that the code for this project has been partially written by Claude Code.