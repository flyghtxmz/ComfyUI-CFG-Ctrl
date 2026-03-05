# ComfyUI-CFG-Ctrl

> **SMC-CFG** — Sliding Mode Control Classifier-Free Guidance for ComfyUI  
> Based on [CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2603.03281) (CVPR 2026)

---

## What is it?

**CFG-Ctrl** reinterprets Classifier-Free Guidance (CFG) through the lens of control theory:

| Method | Description | Formula |
|---|---|---|
| Vanilla CFG | Proportional controller (P-control) with fixed gain | `v = v_u + w * e` |
| **SMC-CFG** | Nonlinear sliding mode controller | `v = v_u + w * e + u_sw` |

Where:
- `e_t = v_cond - v_uncond` — guidance error (discrepancy between conditioned and unconditioned predictions)
- `s_t = e_t + (λ - 1) * e_{t-1}` — exponential sliding surface
- `u_sw = -K * sign(s_t)` — nonlinear switching correction

**Why it matters:** standard CFG at high scales causes instability and semantic overshooting. SMC-CFG adds a correction term that enforces convergence along a sliding manifold, preventing those failures with Lyapunov-guaranteed finite-time stability.

## Installation

Clone this repo into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-CFG-Ctrl.git
```

No extra dependencies required — uses only `torch`, which ComfyUI already ships with.

## Nodes

### `SMC-CFG (CFG-Ctrl)` — Simple

Plug-and-play node. Connect between the model loader and the KSampler.

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Any ComfyUI model |
| `smc_lambda` | float | `5.0` | Decay rate of the sliding surface (λ) |
| `smc_k` | float | `0.2` | Switching gain K |
| `warmup_steps` | int | `2` | Initial steps using vanilla CFG before SMC activates |

### `SMC-CFG Advanced (CFG-Ctrl)` — Advanced

All parameters from the simple node, plus:

| Input | Type | Default | Description |
|---|---|---|---|
| `switching_mode` | enum | `hard` | `hard` = sign(s) (paper default) · `smooth` = tanh(s/δ) |
| `tanh_delta` | float | `0.1` | Smoothing bandwidth δ (only for smooth mode) |
| `normalize_error` | bool | `False` | Normalise the sliding surface to unit norm before applying K (makes behaviour more consistent across models) |
| `smc_end_step` | int | `0` | Deactivate SMC after this many steps (`0` = always active) |

## Workflow

```
[Load Checkpoint] ──► [SMC-CFG (CFG-Ctrl)] ──► [KSampler]
                                                     ▲
                         [CLIP Text Encode] ─────────┤ (positive)
                         [CLIP Text Encode] ─────────┤ (negative)
                         [VAE] ──────────────────────┘ (latent)
```

The **CFG scale on the KSampler** is the proportional gain `w`. SMC adds the nonlinear correction on top — it does not replace your CFG setting.

## Recommended Settings

| Model | CFG Scale | λ (lambda) | K | Notes |
|---|---|---|---|---|
| FLUX.1-dev | 2–3 | 5.0 | 0.2 | Use `warmup_steps = 2` |
| SD3 / SD3.5 | 7.5 | 5.0 | 0.2 | |
| Qwen-Image | 4.0 | 5.0 | 0.2 | |
| Wan Video | 5.0 | 5.0 | 0.2 | |

> **Note:** Models that run at CFG = 1 (fully distilled, e.g. some Flux variants) will see no effect, since `e_t = 0` in those cases.

## How the State Reset Works

The node tracks `e_prev` (error from the previous denoising step) across steps within the same generation. It automatically resets at the start of each new generation by detecting:

1. **Sigma increase** — the scheduler jumped back to high noise (new run started).
2. **Shape change** — the latent tensor dimensions changed (different batch/resolution).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Images look the same as without the node | Make sure CFG scale > 1 on KSampler |
| Artifacts / over-saturation | Lower `smc_k` (try 0.05–0.1) |
| Output too similar to negative prompt | Raise `smc_lambda` |
| Chattering / noisy edges | Switch to `smooth` mode and set `tanh_delta = 0.2` |

## Citation

```bibtex
@misc{wang2026cfgctrlcontrolbasedclassifierfreediffusion,
  title   = {CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance},
  author  = {Hanyang Wang and Yiyang Liu and Jiawei Chi and Fangfu Liu and Ran Xue and Yueqi Duan},
  year    = {2026},
  eprint  = {2603.03281},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2603.03281},
}
```

## License

MIT — do whatever you want, credit appreciated.
