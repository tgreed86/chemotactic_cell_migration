"""Physics-derived input features for chemotaxis GNNs.

The prescribed-chemoattractant model contains the conservative advection term

    -div(n * chi * grad(c)).

This module evaluates a graph-local finite-volume approximation of that term
from the current density and exposes it as an optional node feature.  Keeping
the calculation in PyTorch lets training and autoregressive rollout construct
the same feature from either ground-truth or predicted states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


INPUT_FORM_RATE = "rate"
INPUT_FORM_DELTA = "delta"
INPUT_FORMS = {INPUT_FORM_RATE, INPUT_FORM_DELTA}

ADVECTION_SCHEME_UPWIND = "upwind"
ADVECTION_SCHEME_RUSANOV = "rusanov"
ADVECTION_SCHEME_CENTRAL = "central"
ADVECTION_SCHEMES = {
    ADVECTION_SCHEME_UPWIND,
    ADVECTION_SCHEME_RUSANOV,
    ADVECTION_SCHEME_CENTRAL,
}


@dataclass(frozen=True)
class PhysicsInputConfig:
    """Runtime controls for the optional chemotactic-advection channel."""

    enabled: bool = False
    include_adv: bool = True
    input_form: str = INPUT_FORM_RATE
    advection_scheme: str = ADVECTION_SCHEME_UPWIND
    adv_all_steps: bool = True
    input_weighted: bool = False
    adv_weight: float = 1.0
    normalize_adv_to_minus1_1: bool = False
    adv_min: float | None = None
    adv_max: float | None = None
    normalize_adv_clip: bool = False
    detach_inputs: bool = True


def resolve_physics_input_cfg(cfg: Mapping[str, Any]) -> PhysicsInputConfig:
    """Resolve ``physics_inputs`` using the 1D traffic config conventions."""
    raw = cfg.get("physics_inputs", {})
    if raw is None:
        physics: Mapping[str, Any] = {}
    elif isinstance(raw, Mapping):
        physics = raw
    else:
        raise ValueError("physics_inputs must be a JSON object.")

    input_form = str(physics.get("input_form", INPUT_FORM_RATE)).strip().lower()
    if input_form not in INPUT_FORMS:
        raise ValueError(
            "physics_inputs.input_form must be one of "
            f"{sorted(INPUT_FORMS)}, got {input_form!r}."
        )

    scheme = str(
        physics.get("advection_scheme", ADVECTION_SCHEME_UPWIND)
    ).strip().lower()
    # For this scalar linear face flux, the Godunov flux is the usual upwind
    # flux.  Accepting both spellings makes traffic configs portable.
    if scheme == "godunov":
        scheme = ADVECTION_SCHEME_UPWIND
    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            "physics_inputs.advection_scheme must be one of "
            f"{sorted(ADVECTION_SCHEMES)} (or 'godunov'), got {scheme!r}."
        )

    adv_min_raw = physics.get("adv_min")
    adv_max_raw = physics.get("adv_max")
    adv_min = None if adv_min_raw is None else float(adv_min_raw)
    adv_max = None if adv_max_raw is None else float(adv_max_raw)
    if (adv_min is None) != (adv_max is None):
        raise ValueError(
            "physics_inputs.adv_min and adv_max must either both be supplied "
            "or both be null."
        )
    if adv_min is not None and adv_max is not None and adv_max <= adv_min:
        raise ValueError("physics_inputs.adv_max must be greater than adv_min.")

    adv_weight = float(physics.get("adv_weight", 1.0))
    if not np.isfinite(adv_weight):
        raise ValueError("physics_inputs.adv_weight must be finite.")

    return PhysicsInputConfig(
        enabled=bool(physics.get("enabled", False)),
        include_adv=bool(physics.get("include_adv", True)),
        input_form=input_form,
        advection_scheme=scheme,
        adv_all_steps=bool(physics.get("adv_all_steps", True)),
        input_weighted=bool(physics.get("input_weighted", False)),
        adv_weight=adv_weight,
        normalize_adv_to_minus1_1=bool(
            physics.get(
                "normalize_adv_to_minus1_1",
                physics.get("normalize_adv", False),
            )
        ),
        adv_min=adv_min,
        adv_max=adv_max,
        normalize_adv_clip=bool(physics.get("normalize_adv_clip", False)),
        detach_inputs=bool(physics.get("detach_inputs", True)),
    )


def count_physics_input_channels(physics_cfg: PhysicsInputConfig) -> int:
    """Return the number of enabled per-node physics channels."""
    return int(physics_cfg.enabled and physics_cfg.include_adv)


def face_drift_speed_from_archive(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return chi*grad(c).normal on each stored oriented interior face.

    New archives store the exact face-centered values used by the coarse
    finite-volume operator.  For older archives, the same values are rebuilt
    from the saved Gaussian chemoattractant parameters when possible.  A
    center-interpolated velocity fallback keeps still older compatible
    archives usable.
    """
    edges = np.asarray(data["undirected_edge_index"], dtype=np.int64)
    if edges.ndim != 3 or edges.shape[1] != 2:
        raise ValueError(
            "undirected_edge_index must have shape [trajectory, 2, face]."
        )
    num_trajectories, _, num_faces = edges.shape
    expected = (num_trajectories, num_faces)
    if "face_drift_speed" in data:
        speed = np.asarray(data["face_drift_speed"], dtype=np.float64)
        if speed.shape == (*expected, 1):
            speed = speed[..., 0]
        if speed.shape != expected:
            raise ValueError(
                f"face_drift_speed has shape {speed.shape}; expected {expected}."
            )
    else:
        exact_keys = (
            "shared_face_midpoints",
            "shared_face_normals",
            "chemo_source_count",
            "chemo_source_centers",
            "chemo_source_amplitudes",
            "chemo_source_sigmas",
            "trajectory_chi",
        )
        if all(name in data for name in exact_keys):
            midpoints = np.asarray(data["shared_face_midpoints"], dtype=np.float64)
            normals = np.asarray(data["shared_face_normals"], dtype=np.float64)
            if midpoints.shape != (*expected, 2) or normals.shape != (*expected, 2):
                raise ValueError(
                    "shared_face_midpoints and shared_face_normals must have "
                    "shape [trajectory, face, 2]."
                )
            chi = np.asarray(data["trajectory_chi"], dtype=np.float64)
            speed = np.empty(expected, dtype=np.float64)
            for trajectory in range(num_trajectories):
                count = int(np.asarray(data["chemo_source_count"])[trajectory])
                centers = np.asarray(
                    data["chemo_source_centers"], dtype=np.float64
                )[trajectory, :count]
                amplitudes = np.asarray(
                    data["chemo_source_amplitudes"], dtype=np.float64
                )[trajectory, :count]
                sigmas = np.asarray(
                    data["chemo_source_sigmas"], dtype=np.float64
                )[trajectory, :count]
                delta = midpoints[trajectory, :, None, :] - centers[None, :, :]
                sigma2 = np.square(sigmas)
                exponent = (
                    -0.5 * np.sum(np.square(delta), axis=-1) / sigma2[None, :]
                )
                components = amplitudes[None, :] * np.exp(exponent)
                gradient = np.sum(
                    -components[..., None]
                    * delta
                    / sigma2[None, :, None],
                    axis=1,
                )
                speed[trajectory] = chi[trajectory] * np.sum(
                    gradient * normals[trajectory], axis=-1
                )
        else:
            required = ("drift_velocity", "shared_face_normals")
            missing = [name for name in required if name not in data]
            if missing:
                raise KeyError(
                    "Advection inputs require face_drift_speed or these arrays: "
                    + ", ".join(missing)
                )
            velocity = np.asarray(data["drift_velocity"], dtype=np.float64)
            normals = np.asarray(data["shared_face_normals"], dtype=np.float64)
            num_nodes = velocity.shape[1] if velocity.ndim == 3 else -1
            if velocity.shape != (num_trajectories, num_nodes, 2):
                raise ValueError(
                    "drift_velocity must have shape [trajectory, node, 2]."
                )
            if normals.shape != (*expected, 2):
                raise ValueError(
                    "shared_face_normals must have shape [trajectory, face, 2]."
                )
            trajectory = np.arange(num_trajectories, dtype=np.int64)[:, None]
            source_velocity = velocity[trajectory, edges[:, 0]]
            target_velocity = velocity[trajectory, edges[:, 1]]
            speed = np.sum(
                0.5 * (source_velocity + target_velocity) * normals, axis=-1
            )
    if not np.all(np.isfinite(speed)):
        raise ValueError("face_drift_speed contains non-finite values.")
    return speed.astype(np.float32)[..., None]


def compute_chemotaxis_advection_term_2d(
    *,
    x_state: Tensor,
    undirected_edge_index: Tensor,
    face_drift_speed: Tensor,
    shared_face_length: Tensor,
    cell_area: Tensor,
    advection_scheme: str = ADVECTION_SCHEME_UPWIND,
    validate_inputs: bool = True,
) -> Tensor:
    """Compute a conservative finite-volume chemotactic density rate."""
    if x_state.ndim != 2 or x_state.shape[1] != 1:
        raise ValueError(
            f"x_state must have shape [node, 1], got {tuple(x_state.shape)}."
        )
    if undirected_edge_index.ndim != 2 or undirected_edge_index.shape[0] != 2:
        raise ValueError(
            "undirected_edge_index must have shape [2, face], got "
            f"{tuple(undirected_edge_index.shape)}."
        )
    if undirected_edge_index.dtype != torch.long:
        raise ValueError("undirected_edge_index must have torch.long dtype.")
    undirected_edge_index = undirected_edge_index.to(device=x_state.device)
    num_faces = int(undirected_edge_index.shape[1])

    def face_column(value: Tensor, name: str) -> Tensor:
        result = value.to(device=x_state.device, dtype=x_state.dtype)
        if result.ndim == 1:
            result = result.unsqueeze(-1)
        if result.shape != (num_faces, 1):
            raise ValueError(
                f"{name} must have shape [face, 1], got {tuple(result.shape)}."
            )
        return result

    speed = face_column(face_drift_speed, "face_drift_speed")
    length = face_column(shared_face_length, "shared_face_length")
    area = cell_area.to(device=x_state.device, dtype=x_state.dtype)
    if area.ndim == 1:
        area = area.unsqueeze(-1)
    if area.shape != x_state.shape:
        raise ValueError(
            "cell_area must have shape "
            f"{tuple(x_state.shape)}, got {tuple(area.shape)}."
        )

    scheme = str(advection_scheme).strip().lower()
    if scheme == "godunov":
        scheme = ADVECTION_SCHEME_UPWIND
    if scheme not in ADVECTION_SCHEMES:
        raise ValueError(
            f"advection_scheme must be one of {sorted(ADVECTION_SCHEMES)}."
        )
    if validate_inputs:
        if undirected_edge_index.numel() and (
            int(torch.min(undirected_edge_index)) < 0
            or int(torch.max(undirected_edge_index)) >= x_state.shape[0]
        ):
            raise ValueError("undirected_edge_index contains an invalid node index.")
        if not bool(torch.all(torch.isfinite(speed))):
            raise ValueError("face_drift_speed contains non-finite values.")
        if not bool(torch.all(torch.isfinite(length))) or bool(
            torch.any(length <= 0.0)
        ):
            raise ValueError("shared_face_length values must be finite and positive.")
        if not bool(torch.all(torch.isfinite(area))) or bool(
            torch.any(area <= 0.0)
        ):
            raise ValueError("cell_area values must be finite and positive.")

    sources, targets = undirected_edge_index
    density_source = x_state[sources]
    density_target = x_state[targets]
    if scheme == ADVECTION_SCHEME_UPWIND:
        face_density = torch.where(speed >= 0.0, density_source, density_target)
        face_mass_flux = length * speed * face_density
    else:
        face_mass_flux = length * speed * 0.5 * (
            density_source + density_target
        )
        if scheme == ADVECTION_SCHEME_RUSANOV:
            face_mass_flux = face_mass_flux - 0.5 * length * torch.abs(speed) * (
                density_target - density_source
            )

    amount_rate = torch.zeros_like(x_state)
    amount_rate.index_add_(0, sources, -face_mass_flux)
    amount_rate.index_add_(0, targets, face_mass_flux)
    return amount_rate / area


def compute_adv_input_channel_2d(
    *,
    x_state: Tensor,
    undirected_edge_index: Tensor,
    face_drift_speed: Tensor,
    shared_face_length: Tensor,
    cell_area: Tensor,
    dt_node: Tensor,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
    validate_inputs: bool = True,
) -> Tensor:
    """Compute the configured advection channel before normalization."""
    should_compute = physics_cfg.adv_all_steps or int(step_k) == 0
    if should_compute:
        advection = compute_chemotaxis_advection_term_2d(
            x_state=x_state,
            undirected_edge_index=undirected_edge_index,
            face_drift_speed=face_drift_speed,
            shared_face_length=shared_face_length,
            cell_area=cell_area,
            advection_scheme=physics_cfg.advection_scheme,
            validate_inputs=validate_inputs,
        )
    else:
        advection = torch.zeros_like(x_state)
    if physics_cfg.input_weighted:
        advection = float(physics_cfg.adv_weight) * advection
    if physics_cfg.input_form == INPUT_FORM_DELTA:
        dt = dt_node.to(device=x_state.device, dtype=x_state.dtype)
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)
        if dt.shape != x_state.shape:
            raise ValueError(
                "dt_node must have shape "
                f"{tuple(x_state.shape)}, got {tuple(dt.shape)}."
            )
        advection = dt * advection
    return advection


def _normalize_adv_to_minus1_1(
    advection: Tensor,
    *,
    physics_cfg: PhysicsInputConfig,
    eps: float = 1.0e-12,
) -> Tensor:
    if physics_cfg.adv_min is None or physics_cfg.adv_max is None:
        raise ValueError(
            "Advection normalization is enabled but adv_min/adv_max are missing."
        )
    denominator = physics_cfg.adv_max - physics_cfg.adv_min
    if abs(denominator) <= eps:
        normalized = torch.zeros_like(advection)
    else:
        normalized = 2.0 * (advection - physics_cfg.adv_min) / denominator - 1.0
    if physics_cfg.normalize_adv_clip:
        normalized = torch.clamp(normalized, min=-1.0, max=1.0)
    return normalized


def build_physics_augmented_inputs_2d(
    *,
    x_base: Tensor,
    x_state: Tensor,
    undirected_edge_index: Tensor,
    face_drift_speed: Tensor,
    shared_face_length: Tensor,
    cell_area: Tensor,
    dt_node: Tensor,
    physics_cfg: PhysicsInputConfig,
    step_k: int,
    validate_inputs: bool = True,
) -> Tensor:
    """Append the configured chemotactic-advection channel to base features."""
    if x_base.ndim != 2:
        raise ValueError(
            f"x_base must have shape [node, feature], got {tuple(x_base.shape)}."
        )
    if x_state.ndim != 2 or x_state.shape[1] != 1:
        raise ValueError(
            f"x_state must have shape [node, 1], got {tuple(x_state.shape)}."
        )
    if x_base.shape[0] != x_state.shape[0]:
        raise ValueError("x_base and x_state must have matching node counts.")
    if not physics_cfg.enabled or not physics_cfg.include_adv:
        return x_base

    computed = physics_cfg.adv_all_steps or int(step_k) == 0
    advection = compute_adv_input_channel_2d(
        x_state=x_state,
        undirected_edge_index=undirected_edge_index,
        face_drift_speed=face_drift_speed,
        shared_face_length=shared_face_length,
        cell_area=cell_area,
        dt_node=dt_node,
        physics_cfg=physics_cfg,
        step_k=step_k,
        validate_inputs=validate_inputs,
    )
    # Keep gated-off steps neutral even when the fitted range is asymmetric.
    if physics_cfg.normalize_adv_to_minus1_1 and computed:
        advection = _normalize_adv_to_minus1_1(
            advection, physics_cfg=physics_cfg
        )
    if physics_cfg.detach_inputs:
        advection = advection.detach()
    return torch.cat((x_base, advection), dim=-1)
