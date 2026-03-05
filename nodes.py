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
    u_sw = -K * φ(s_t)
    where φ is either hard sign or smooth tanh(s_t / δ)

Corrected error + guidance (u_sw is scaled by w, matching original):
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

    'hard'  → sign(s)            – original paper formulation
    'smooth' → tanh(s / δ)       – differentiable approximation; reduces
                                   chattering at the cost of a small boundary
                                   layer around s = 0
    """
    if mode == "smooth":
        return torch.tanh(s / (tanh_delta + 1e-8))
    return torch.sign(s)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – Simple
# ─────────────────────────────────────────────────────────────────────────────

class SMCCFGNode:
    """
    Patches a MODEL to apply SMC-CFG guidance.

    Plug-and-play: connect between any checkpoint loader and the KSampler.
    The CFG scale on the KSampler acts as the proportional gain w.
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
                            "Controls how strongly past error influences correction. "
                            "Default: 0.05. Recommended by paper: 5.0 for FLUX / SD3 / Wan."
                        ),
                    },
                ),
                "smc_k": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": (
                            "Switching gain K. Scales the nonlinear correction term. "
                            "Too high → over-correction; too low → same as vanilla CFG. "
                            "Default: 0.3. Recommended: 0.2"
                        ),
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": (
                            "Steps at the start of denoising where NO CFG is applied: "
                            "the sampler uses only the conditional prediction. "
                            "SMC initialises at the first step after warmup ends."
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
        "from CFG-Ctrl (CVPR 2026). Stabilises guidance at large CFG scales "
        "via a nonlinear correction term. "
        "Supports any flow-based model: FLUX, SD3/SD3.5, Wan Video, etc."
    )

    def patch(
        self,
        model,
        smc_lambda: float,
        smc_k: float,
        warmup_steps: int,
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

            # On the first SMC step initialise e_prev with the current e_t,
            # then apply SMC immediately (s_t = λ * e_t on this step).
            if state.e_prev is None:
                state.e_prev = e_t.detach().clone()

            s_t         = (e_t - state.e_prev) + smc_lambda * state.e_prev
            u_sw        = -smc_k * torch.sign(s_t)
            corrected_e = e_t + u_sw

            # Store the corrected error (matches original implementation).
            state.e_prev = corrected_e.detach().clone()
            # u_sw is implicitly scaled by w here, matching the original paper.
            return uncond_pred + w * corrected_e

        m.set_model_sampler_cfg_function(cfg_fn)
        return (m,)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – Advanced
# ─────────────────────────────────────────────────────────────────────────────

class SMCCFGAdvancedNode:
    """
    Advanced version with extra tuning options:

    • switching_mode – choose between hard sign (paper default) and smooth tanh
    • tanh_delta     – smoothing bandwidth for the tanh approximation
    • normalize_error – normalise the sliding surface s_t to unit norm before
                        applying the gain, making K scale-invariant across models
    • cfg_end_step   – turn off SMC after this many steps (set 0 to always apply)
    """

    SWITCHING_MODES = ["hard", "smooth"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "smc_lambda": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 50.0, "step": 0.01},
                ),
                "smc_k": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "warmup_steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "switching_mode": (
                    cls.SWITCHING_MODES,
                    {
                        "default": "hard",
                        "tooltip": (
                            "'hard' = sign(s) – exact paper formulation. "
                            "'smooth' = tanh(s/δ) – differentiable, reduces chattering."
                        ),
                    },
                ),
                "tanh_delta": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 1e-4,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": (
                            "Smoothing bandwidth δ for the tanh switching function. "
                            "Smaller = closer to hard sign. Only used when switching_mode='smooth'."
                        ),
                    },
                ),
                "normalize_error": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Normalize the sliding surface s_t to unit norm before "
                            "applying gain K. Makes the node behaviour more consistent "
                            "across models with different activation scales."
                        ),
                    },
                ),
                "smc_end_step": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": (
                            "Deactivate SMC after this many steps (fall back to vanilla CFG). "
                            "0 = always active."
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
        "Advanced SMC-CFG node with additional tuning controls: "
        "smooth switching function, error normalisation, and per-step gating."
    )

    def patch(
        self,
        model,
        smc_lambda: float,
        smc_k: float,
        warmup_steps: int,
        switching_mode: str,
        tanh_delta: float,
        normalize_error: bool,
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
                # On the first SMC step initialise e_prev, then apply immediately.
                if state.e_prev is None:
                    state.e_prev = e_t.detach().clone()

                s_t = (e_t - state.e_prev) + smc_lambda * state.e_prev

                if normalize_error:
                    norm = s_t.norm() + 1e-8
                    s_t  = s_t / norm

                phi         = _switching_fn(s_t, switching_mode, tanh_delta)
                u_sw        = -smc_k * phi
                corrected_e = e_t + u_sw

                # Store corrected error (matches original implementation).
                state.e_prev = corrected_e.detach().clone()
                return uncond_pred + w * corrected_e

            # SMC gated off (smc_end_step reached): fall back to vanilla CFG.
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
