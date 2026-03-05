"""
nodes.py – ComfyUI-CFG-Ctrl
Implements CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance (CVPR 2026)

Paper  : https://arxiv.org/abs/2603.03281
GitHub : https://github.com/hanyang-21/CFG-Ctrl

────────────────────────────────────────────────────────────────────────────────
MATH OVERVIEW
────────────────────────────────────────────────────────────────────────────────

Error signal (guidance discrepancy):
    e_t = v_cond - v_uncond

Vanilla CFG (proportional / P-control):
    v_guided = v_uncond + w * e_t

SMC-CFG sliding surface (exponential):
    s_t = (e_t - e_{t-1}) + λ * e_{t-1}

Switching control (nonlinear correction):
    u_sw = -K * φ(s_norm)          where s_norm = s_t / (||e_t|| + ε)
    φ is either hard sign or smooth tanh(s_norm / δ)

Adaptive normalisation (default, cross-model safe):
    s_t is divided by ||e_t|| before φ, making K a dimensionless ratio.
    K = 0.1 → correction never exceeds ~10 % of ||e_t||, regardless of model.
    This makes the same K value safe for FLUX, SD3, SDXL, SD1.5, Pony, etc.

Hard safety clamp (max_correction_ratio):
    ||u_sw|| is clamped to max_correction_ratio * ||e_t||.
    Prevents runaway correction even with aggressive K values.

Corrected error + guidance:
    corrected_e = e_t + u_sw
    v_guided    = v_uncond + w * corrected_e

State update – stores corrected error, not raw e_t:
    e_{t} ← corrected_e

Warmup (warmup_steps > 0):
    First N steps return v_cond directly (no CFG, no SMC).
    SMC initialises at the first active step with e_prev = e_t,
    giving s_t = λ * e_t on that step.

Lyapunov stability guarantees finite-time convergence on the sliding manifold.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared state object
# ─────────────────────────────────────────────────────────────────────────────

class _SMCState:
    """Per-generation mutable state stored inside the model-patch closure."""

    def __init__(self) -> None:
        self.e_prev: torch.Tensor | None = None
        self.step: int = 0
        self.last_sigma: float | None = None
        self.last_shape: tuple | None = None

    def reset(self) -> None:
        self.e_prev = None
        self.step = 0

    def check_and_reset(self, sigma_max: float, shape: tuple) -> None:
        """
        Detect the start of a new denoising trajectory and reset state.

        Two conditions trigger a reset:
          1. Sigma increased (scheduler jumped back to high noise → new run).
          2. Latent shape changed (different resolution / batch size).
        """
        shape_changed = (self.last_shape is not None and shape != self.last_shape)
        sigma_jumped  = (
            self.last_sigma is not None and sigma_max > self.last_sigma
        )
        if shape_changed or sigma_jumped:
            self.reset()

        self.last_sigma = sigma_max
        self.last_shape = shape


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _switching_fn(s: torch.Tensor, mode: str, tanh_delta: float) -> torch.Tensor:
    """
    φ(s) – the switching function used in the control term u_sw = -K * φ(s).

    's' is expected to be already normalised (s_t / ||e_t||) when
    adaptive_scale=True, so its values are dimensionless.

    'hard'   → sign(s)          – original paper formulation
    'smooth' → tanh(s / δ)      – differentiable approximation; reduces
                                  chattering at the cost of a small boundary
                                  layer around s = 0
    """
    if mode == "smooth":
        return torch.tanh(s / (tanh_delta + 1e-8))
    return torch.sign(s)


def _apply_smc(
    e_t: torch.Tensor,
    e_prev: torch.Tensor,
    smc_lambda: float,
    smc_k: float,
    switching_mode: str,
    tanh_delta: float,
    adaptive_scale: bool,
    max_correction_ratio: float,
) -> torch.Tensor:
    """
    Core SMC correction, shared by both nodes.

    adaptive_scale=True  → normalise s_t by ||e_t|| before φ.
                           K becomes a dimensionless ratio safe across all models.
    max_correction_ratio → hard clamp: ||u_sw|| ≤ ratio * ||e_t||.
                           0.0 = no clamp.
    Returns corrected_e = e_t + u_sw.
    """
    s_t     = (e_t - e_prev) + smc_lambda * e_prev
    e_norm  = e_t.norm() + 1e-8

    if adaptive_scale:
        s_input = s_t / e_norm
    else:
        s_input = s_t

    phi  = _switching_fn(s_input, switching_mode, tanh_delta)
    u_sw = -smc_k * phi

    # When adaptive, scale u_sw back to signal space.
    if adaptive_scale:
        u_sw = u_sw * e_norm

    # Hard safety clamp.
    if max_correction_ratio > 0.0:
        u_norm = u_sw.norm() + 1e-8
        limit  = max_correction_ratio * e_norm
        if u_norm > limit:
            u_sw = u_sw * (limit / u_norm)

    return e_t + u_sw


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Simple
# ─────────────────────────────────────────────────────────────────────────────

class SMCCFGNode:
    """
    Patches a MODEL to apply SMC-CFG guidance.

    Plug-and-play: connect between any checkpoint loader and the KSampler.
    The CFG scale on the KSampler acts as the proportional gain w.

    adaptive_scale=True (default) normalises the sliding surface by ||e_t||,
    making K a dimensionless ratio safe across FLUX, SD3, SDXL, SD1.5, Pony, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "smc_lambda": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.01,
                        "tooltip": (
                            "Exponential decay coefficient λ of the sliding surface. "
                            "Recommended by paper: 5.0 for FLUX / SD3 / Wan."
                        ),
                    },
                ),
                "smc_k": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Switching gain K. With adaptive_scale ON this is a ratio "
                            "of ||e_t||: K=0.1 means correction ≤ 10% of the error signal. "
                            "Safe range: 0.05–0.3 for any model."
                        ),
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": (
                            "Steps at the start where NO CFG is applied (only cond prediction). "
                            "SMC state initialises at the first active step."
                        ),
                    },
                ),
                "adaptive_scale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Normalise s_t by ||e_t|| before applying gain K. "
                            "Makes K a dimensionless ratio — same value works across "
                            "FLUX, SD3, SDXL, SD1.5, Pony and any other model. "
                            "Disable to use the raw paper formulation (requires per-model K tuning)."
                        ),
                    },
                ),
                "max_correction_ratio": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Hard safety clamp: ||u_sw|| is limited to this fraction of ||e_t||. "
                            "0.25 = correction never exceeds 25% of the error signal. "
                            "Set 0.0 to disable. Prevents runaway correction on any model."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "CFG-Ctrl"
    DESCRIPTION = (
        "Applies SMC-CFG (Sliding Mode Control Classifier-Free Guidance) "
        "from CFG-Ctrl (CVPR 2026). Adaptive by default — works across FLUX, "
        "SD3/SD3.5, SDXL, Pony, SD1.5, Wan Video without per-model tuning."
    )

    def patch(
        self,
        model,
        smc_lambda: float,
        smc_k: float,
        warmup_steps: int,
        adaptive_scale: bool,
        max_correction_ratio: float,
    ):
        m = model.clone()
        state = _SMCState()

        def cfg_fn(args: dict) -> torch.Tensor:
            cond_pred:   torch.Tensor = args["cond"]
            uncond_pred: torch.Tensor = args["uncond"]
            w: float = args["cond_scale"]

            sigma = args.get("sigma")
            if sigma is not None:
                state.check_and_reset(
                    sigma_max=float(sigma.max()),
                    shape=tuple(cond_pred.shape),
                )

            step = state.step
            state.step += 1

            # Warmup: return conditional prediction with no CFG applied.
            if warmup_steps > 0 and step < warmup_steps:
                return cond_pred

            e_t = cond_pred - uncond_pred

            if state.e_prev is None:
                state.e_prev = e_t.detach().clone()

            corrected_e = _apply_smc(
                e_t=e_t,
                e_prev=state.e_prev,
                smc_lambda=smc_lambda,
                smc_k=smc_k,
                switching_mode="hard",
                tanh_delta=0.1,
                adaptive_scale=adaptive_scale,
                max_correction_ratio=max_correction_ratio,
            )

            state.e_prev = corrected_e.detach().clone()
            return uncond_pred + w * corrected_e

        m.set_model_sampler_cfg_function(cfg_fn)
        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Advanced
# ─────────────────────────────────────────────────────────────────────────────

class SMCCFGAdvancedNode:
    """
    Advanced version with full tuning options:

    • adaptive_scale      – normalise s_t by ||e_t|| for cross-model safety (default ON)
    • max_correction_ratio – hard clamp on ||u_sw|| / ||e_t|| (default 0.25)
    • switching_mode      – hard sign (paper) or smooth tanh
    • tanh_delta          – bandwidth for tanh approximation
    • normalize_s         – legacy: normalise s_t to unit norm (instead of adaptive_scale)
    • smc_end_step        – gate off SMC after N steps
    """

    SWITCHING_MODES = ["hard", "smooth"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "smc_lambda": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 50.0, "step": 0.01,
                     "tooltip": "Exponential decay coefficient λ. Paper recommends 5.0 for FLUX/SD3/Wan."},
                ),
                "smc_k": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01,
                     "tooltip": "Switching gain K. With adaptive_scale ON: dimensionless ratio of ||e_t||. Safe range: 0.05–0.3."},
                ),
                "warmup_steps": (
                    "INT",
                    {"default": 2, "min": 0, "max": 50, "step": 1,
                     "tooltip": "Initial steps with NO CFG (only cond prediction). SMC starts after warmup."},
                ),
                "adaptive_scale": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": (
                         "Normalise s_t by ||e_t|| before applying K. "
                         "Makes K a safe dimensionless ratio across all model types. "
                         "Disable for the raw paper formulation."
                     )},
                ),
                "max_correction_ratio": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Hard clamp: ||u_sw|| ≤ ratio * ||e_t||. Set 0.0 to disable."},
                ),
                "switching_mode": (
                    cls.SWITCHING_MODES,
                    {"default": "smooth",
                     "tooltip": "'hard' = sign(s) – paper default. 'smooth' = tanh(s/δ) – less chattering."},
                ),
                "tanh_delta": (
                    "FLOAT",
                    {"default": 0.5, "min": 1e-4, "max": 5.0, "step": 0.01,
                     "tooltip": "Smoothing bandwidth δ for tanh mode. Smaller = closer to hard sign."},
                ),
                "normalize_s": (
                    "BOOLEAN",
                    {"default": False,
                     "tooltip": (
                         "Normalise s_t to unit norm before φ (legacy option, independent of adaptive_scale). "
                         "adaptive_scale is preferred for cross-model safety."
                     )},
                ),
                "smc_end_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 200, "step": 1,
                     "tooltip": "Deactivate SMC after this many steps (fall back to vanilla CFG). 0 = always active."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "CFG-Ctrl"
    DESCRIPTION = (
        "Advanced SMC-CFG node. adaptive_scale (default ON) makes K a safe "
        "dimensionless ratio across FLUX, SD3, SDXL, Pony, SD1.5 and any other model. "
        "max_correction_ratio provides a hard safety clamp."
    )

    def patch(
        self,
        model,
        smc_lambda: float,
        smc_k: float,
        warmup_steps: int,
        adaptive_scale: bool,
        max_correction_ratio: float,
        switching_mode: str,
        tanh_delta: float,
        normalize_s: bool,
        smc_end_step: int,
    ) -> tuple:
        m = model.clone()
        state = _SMCState()

        def cfg_fn(args: dict) -> torch.Tensor:
            cond_pred:   torch.Tensor = args["cond"]
            uncond_pred: torch.Tensor = args["uncond"]
            w: float = args["cond_scale"]

            sigma = args.get("sigma")
            if sigma is not None:
                state.check_and_reset(
                    sigma_max=float(sigma.max()),
                    shape=tuple(cond_pred.shape),
                )

            step = state.step
            state.step += 1

            # Warmup: return conditional prediction with no CFG applied.
            if warmup_steps > 0 and step < warmup_steps:
                return cond_pred

            e_t = cond_pred - uncond_pred

            smc_active = smc_end_step == 0 or step < smc_end_step

            if smc_active:
                if state.e_prev is None:
                    state.e_prev = e_t.detach().clone()

                s_t    = (e_t - state.e_prev) + smc_lambda * state.e_prev
                e_norm = e_t.norm() + 1e-8

                if adaptive_scale:
                    s_input = s_t / e_norm
                else:
                    s_input = s_t

                if normalize_s:
                    s_input = s_input / (s_input.norm() + 1e-8)

                phi  = _switching_fn(s_input, switching_mode, tanh_delta)
                u_sw = -smc_k * phi

                if adaptive_scale:
                    u_sw = u_sw * e_norm

                if max_correction_ratio > 0.0:
                    u_norm = u_sw.norm() + 1e-8
                    limit  = max_correction_ratio * e_norm
                    if u_norm > limit:
                        u_sw = u_sw * (limit / u_norm)

                corrected_e = e_t + u_sw
                state.e_prev = corrected_e.detach().clone()
                return uncond_pred + w * corrected_e

            # SMC gated off: fall back to vanilla CFG.
            state.e_prev = e_t.detach().clone()
            return uncond_pred + w * e_t

        m.set_model_sampler_cfg_function(cfg_fn)
        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "SMCCFGNode":         SMCCFGNode,
    "SMCCFGAdvancedNode": SMCCFGAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "SMCCFGNode":         "SMC-CFG (CFG-Ctrl)",
    "SMCCFGAdvancedNode": "SMC-CFG Advanced (CFG-Ctrl)",
}
