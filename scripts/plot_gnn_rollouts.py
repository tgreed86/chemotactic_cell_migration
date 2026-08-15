#!/usr/bin/env python3
"""Evaluate and visualize held-out autoregressive chemotaxis rollouts.

The script is architecture-agnostic: it reconstructs SAGEConv, MeshGraphNet,
or FluxGraphNet from a checkpoint produced by ``scripts/train.py`` and rolls
the model forward from each held-out test initial condition. It writes:

* ``test_rollout_predictions.npz`` with truth and predictions;
* ``test_rollout_metrics.csv`` and ``summary_metrics.json``;
* a test-set summary PNG; and
* spatial comparison and chemotaxis-diagnostic PNGs for representative test
  trajectories (best, median, and worst rollout RMSE by default).

Example:

    python scripts/plot_gnn_rollouts.py \
        --checkpoint runs/chemotaxis/sageconv_delta_ar1/checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch and PyTorch Geometric. Use the same "
        "environment used by scripts/train.py."
    ) from exc

from train import (
    FluxGraphNet,
    Normalization,
    choose_device,
    decode_next_state,
    load_archive,
    model_from_checkpoint,
    prepare_features,
    project_conservative_target,
)


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load and validate a trusted chemotaxis checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")
    required = (
        "model_state_dict",
        "model_kwargs",
        "normalization",
        "split_trajectory_ids",
        "target_type",
        "dt",
        "data_path",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint is missing fields: {missing}")
    if str(checkpoint["target_type"]) not in ("state", "delta", "rate"):
        raise ValueError("Checkpoint target_type is invalid.")
    return checkpoint


def _batched_indices(indices: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Join trajectory-specific graphs into one disjoint PyG-style graph."""
    batch_size = indices.shape[0]
    offsets = np.arange(batch_size, dtype=np.int64)[:, None, None] * num_nodes
    batched = indices + offsets
    return torch.from_numpy(
        np.ascontiguousarray(batched.transpose(1, 0, 2).reshape(2, -1))
    ).long()


@torch.inference_mode()
def predict_rollouts(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    trajectory_ids: np.ndarray,
    *,
    num_steps: int,
    normalization: Normalization,
    target_type: str,
    mass_projection: bool,
    device: torch.device,
    batch_size: int,
    include_chi: bool,
    include_drift_velocity: bool,
    include_absolute_positions: bool,
    include_boundary_distances: bool,
) -> np.ndarray:
    """Autoregressively predict selected trajectory-specific graphs."""
    static, edge_attr, undirected_edge_attr = prepare_features(
        data,
        normalization,
        include_chi=include_chi,
        include_drift_velocity=include_drift_velocity,
        include_absolute_positions=include_absolute_positions,
        include_boundary_distances=include_boundary_distances,
    )
    states = np.asarray(data["rollout_states"], dtype=np.float32)
    edge_indices = np.asarray(data["edge_index"], dtype=np.int64)
    face_indices = np.asarray(data["undirected_edge_index"], dtype=np.int64)
    areas = np.asarray(data["cell_areas"], dtype=np.float32)
    dt = float(np.asarray(data["dt"]).item())
    num_nodes = states.shape[2]
    predictions: List[np.ndarray] = []

    for first in range(0, trajectory_ids.size, batch_size):
        ids = trajectory_ids[first : first + batch_size]
        count = ids.size
        current = torch.from_numpy(states[ids, 0]).to(device=device)
        static_batch = torch.from_numpy(static[ids]).to(device=device)
        directed_index = _batched_indices(edge_indices[ids], num_nodes).to(device)
        face_index = _batched_indices(face_indices[ids], num_nodes).to(device)
        directed_attr = torch.from_numpy(edge_attr[ids]).to(device=device).reshape(
            -1, edge_attr.shape[-1]
        )
        face_attr = torch.from_numpy(undirected_edge_attr[ids]).to(
            device=device
        ).reshape(-1, undirected_edge_attr.shape[-1])
        cell_area = torch.from_numpy(areas[ids, :, None]).to(device=device).reshape(
            -1, 1
        )
        dt_node = torch.full_like(cell_area, dt)
        graph_index = torch.arange(
            count, dtype=torch.long, device=device
        ).repeat_interleave(num_nodes)
        predicted = [current]

        for _ in range(num_steps):
            normalized_density = (
                current - normalization.density_mean
            ) / normalization.density_std
            node_input = torch.cat((normalized_density, static_batch), dim=-1)
            normalized_target = model(
                node_input.reshape(count * num_nodes, -1),
                directed_index,
                edge_attr=directed_attr,
                undirected_edge_index=face_index,
                undirected_edge_attr=face_attr,
                cell_area=cell_area,
                current=current.reshape(-1, 1),
            )
            raw_target = (
                normalized_target * normalization.target_std
                + normalization.target_mean
            )
            if mass_projection and not isinstance(model, FluxGraphNet):
                raw_target = project_conservative_target(
                    raw_target,
                    current=current.reshape(-1, 1),
                    cell_area=cell_area,
                    graph_index=graph_index,
                    num_graphs=count,
                    target_type=target_type,
                )
            next_state = decode_next_state(
                raw_target,
                current.reshape(-1, 1),
                target_type,
                dt_node,
            ).reshape(count, num_nodes, 1)
            predicted.append(next_state)
            current = next_state

        batch_prediction = torch.stack(predicted, dim=1)
        predictions.append(batch_prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def compute_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    cell_areas: np.ndarray,
    chemo: np.ndarray,
    centers: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute spatial error, conservation, and chemotaxis diagnostics."""
    error = prediction - truth
    persistence = np.broadcast_to(truth[:, :1], truth.shape)
    persistence_error = persistence - truth
    model_rmse_by_time = np.sqrt(np.mean(error[..., 0] ** 2, axis=2))
    persistence_rmse_by_time = np.sqrt(
        np.mean(persistence_error[..., 0] ** 2, axis=2)
    )
    model_mae_by_time = np.mean(np.abs(error[..., 0]), axis=2)
    future_error = error[:, 1:, :, 0]
    future_persistence_error = persistence_error[:, 1:, :, 0]

    truth_scalar = truth[..., 0]
    prediction_scalar = prediction[..., 0]
    truth_mass = np.sum(truth_scalar * cell_areas[:, None, :], axis=2)
    prediction_mass = np.sum(
        prediction_scalar * cell_areas[:, None, :], axis=2
    )
    mass_error = prediction_mass - truth_mass
    negative_fraction = np.mean(prediction_scalar < 0.0, axis=2)

    def weighted_chemo(states: np.ndarray) -> np.ndarray:
        weights = states * cell_areas[:, None, :]
        return np.sum(weights * chemo[:, None, :], axis=2) / np.maximum(
            np.sum(weights, axis=2), np.finfo(np.float64).eps
        )

    def center_of_mass(states: np.ndarray) -> np.ndarray:
        weights = states * cell_areas[:, None, :]
        return np.sum(weights[..., None] * centers[:, None, :, :], axis=2) / np.maximum(
            np.sum(weights, axis=2)[..., None], np.finfo(np.float64).eps
        )

    truth_com = center_of_mass(truth_scalar)
    prediction_com = center_of_mass(prediction_scalar)
    return {
        "persistence": persistence,
        "model_rmse_by_time": model_rmse_by_time,
        "persistence_rmse_by_time": persistence_rmse_by_time,
        "model_mae_by_time": model_mae_by_time,
        "model_rollout_rmse": np.sqrt(np.mean(future_error**2, axis=(1, 2))),
        "persistence_rollout_rmse": np.sqrt(
            np.mean(future_persistence_error**2, axis=(1, 2))
        ),
        "final_step_rmse": model_rmse_by_time[:, -1],
        "truth_mass": truth_mass,
        "prediction_mass": prediction_mass,
        "mass_error": mass_error,
        "negative_fraction": negative_fraction,
        "truth_weighted_chemo": weighted_chemo(truth_scalar),
        "prediction_weighted_chemo": weighted_chemo(prediction_scalar),
        "truth_com": truth_com,
        "prediction_com": prediction_com,
        "truth_com_displacement": np.linalg.norm(
            truth_com - truth_com[:, :1], axis=2
        ),
        "prediction_com_displacement": np.linalg.norm(
            prediction_com - prediction_com[:, :1], axis=2
        ),
    }


def representative_trajectories(
    trajectory_ids: np.ndarray, rollout_rmse: np.ndarray
) -> List[int]:
    """Return distinct best, median, and worst held-out trajectory IDs."""
    order = np.argsort(rollout_rmse)
    positions = (0, order.size // 2, order.size - 1)
    return list(dict.fromkeys(int(trajectory_ids[order[position]]) for position in positions))


def validate_selected(
    requested: Sequence[int], test_ids: np.ndarray
) -> List[int]:
    allowed = set(int(value) for value in test_ids)
    invalid = sorted(set(int(value) for value in requested) - allowed)
    if invalid:
        raise ValueError(
            "Requested trajectories are not in the checkpoint test split: "
            + ", ".join(map(str, invalid))
        )
    return list(dict.fromkeys(int(value) for value in requested))


def default_snapshot_steps(num_times: int) -> List[int]:
    return sorted(set(np.rint(np.linspace(0, num_times - 1, 4)).astype(int).tolist()))


def validate_snapshot_steps(steps: Sequence[int], num_times: int) -> List[int]:
    values = list(dict.fromkeys(int(step) for step in steps))
    if not values or min(values) < 0 or max(values) >= num_times:
        raise ValueError(f"snapshot steps must be in [0, {num_times - 1}].")
    return values


def triangulation(data: Dict[str, np.ndarray], trajectory: int) -> mtri.Triangulation:
    vertices = data["mesh_vertices"][trajectory]
    triangles = data["triangles"][trajectory]
    return mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)


def cell_center_triangulation(
    data: Dict[str, np.ndarray], trajectory: int
) -> mtri.Triangulation:
    """Triangulate cell centers for contouring cell-averaged fields."""
    centers = np.asarray(data["pos"][trajectory], dtype=np.float64)
    return mtri.Triangulation(centers[:, 0], centers[:, 1])


def overlay_chemoattractant_contours(
    axis: plt.Axes,
    contour_mesh: mtri.Triangulation,
    chemoattractant: np.ndarray,
    *,
    num_levels: int,
    alpha: float,
) -> None:
    """Overlay high-contrast contours of the static chemoattractant field."""
    lower = float(np.min(chemoattractant))
    upper = float(np.max(chemoattractant))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return
    levels = np.linspace(lower, upper, num_levels + 2, dtype=np.float64)[1:-1]
    # A dark underlay keeps the white contours visible on both light and dark
    # regions of the density colormap.
    axis.tricontour(
        contour_mesh,
        chemoattractant,
        levels=levels,
        colors="black",
        linewidths=1.5,
        alpha=0.45 * alpha,
        zorder=3,
    )
    axis.tricontour(
        contour_mesh,
        chemoattractant,
        levels=levels,
        colors="white",
        linewidths=0.75,
        alpha=alpha,
        zorder=4,
    )


def _style_spatial_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(0.0, float(np.asarray(axis.get_xlim()).max()))
    axis.set_xticks([])
    axis.set_yticks([])


def plot_spatial_comparison(
    data: Dict[str, np.ndarray],
    truth: np.ndarray,
    prediction: np.ndarray,
    trajectory: int,
    output_path: Path,
    *,
    snapshot_steps: Sequence[int],
    dt: float,
    model_label: str,
    dpi: int,
    cmap: str,
    chemo_contours: bool,
    chemo_contour_levels: int,
    chemo_contour_alpha: float,
) -> None:
    """Plot truth, prediction, error, and optional chemoattractant contours."""
    mesh = triangulation(data, trajectory)
    contour_mesh = None
    chemoattractant = None
    if chemo_contours:
        contour_mesh = cell_center_triangulation(data, trajectory)
        chemoattractant = np.asarray(
            data["chemoattractant"][trajectory, :, 0], dtype=np.float64
        )
    error = prediction - truth
    steps = np.asarray(snapshot_steps, dtype=np.int64)
    density_min = float(min(np.min(truth[steps]), np.min(prediction[steps])))
    density_max = float(max(np.max(truth[steps]), np.max(prediction[steps])))
    error_limit = max(float(np.max(np.abs(error[steps]))), 1.0e-10)
    figure, axes = plt.subplots(
        len(steps), 3, figsize=(11.5, 3.15 * len(steps)), squeeze=False,
        constrained_layout=True,
    )
    density_artist = None
    error_artist = None
    for row, step in enumerate(steps):
        for column, (values, color_map, lower, upper) in enumerate(
            (
                (truth[step], cmap, density_min, density_max),
                (prediction[step], cmap, density_min, density_max),
                (error[step], "coolwarm", -error_limit, error_limit),
            )
        ):
            artist = axes[row, column].tripcolor(
                mesh,
                facecolors=values,
                shading="flat",
                cmap=color_map,
                vmin=lower,
                vmax=upper,
                edgecolors="none",
            )
            if column < 2 and contour_mesh is not None and chemoattractant is not None:
                overlay_chemoattractant_contours(
                    axes[row, column],
                    contour_mesh,
                    chemoattractant,
                    num_levels=chemo_contour_levels,
                    alpha=chemo_contour_alpha,
                )
            _style_spatial_axis(axes[row, column])
            if column < 2:
                density_artist = artist
            else:
                error_artist = artist
        axes[row, 0].set_ylabel(f"step {step}\nt={step * dt:.4g}")
    axes[0, 0].set_title("Ground truth")
    axes[0, 1].set_title(f"{model_label} prediction")
    axes[0, 2].set_title("Prediction − truth")
    if density_artist is not None:
        colorbar = figure.colorbar(
            density_artist, ax=list(axes[:, :2].ravel()), shrink=0.86, pad=0.015
        )
        colorbar.set_label("Cell density")
    if error_artist is not None:
        colorbar = figure.colorbar(
            error_artist, ax=list(axes[:, 2]), shrink=0.86, pad=0.015
        )
        colorbar.set_label("Density error")
    title = f"Held-out trajectory {trajectory}: autoregressive rollout"
    if chemo_contours:
        title += "\nWhite contours show chemoattractant concentration c"
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_trajectory_diagnostics(
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: Dict[str, np.ndarray],
    local_index: int,
    trajectory: int,
    output_path: Path,
    *,
    dt: float,
    model_label: str,
    dpi: int,
) -> None:
    """Plot error, conservation, bounds, and biological summary quantities."""
    times = np.arange(truth.shape[0], dtype=np.float64) * dt
    figure, axes = plt.subplots(2, 3, figsize=(14.3, 8.3), constrained_layout=True)
    ax_error, ax_mass, ax_bounds, ax_chemo, ax_com, ax_com_path = axes.ravel()

    ax_error.plot(
        times, metrics["model_rmse_by_time"][local_index], label=model_label
    )
    ax_error.plot(
        times,
        metrics["persistence_rmse_by_time"][local_index],
        color="0.45",
        label="Persistence",
    )
    ax_error.set_title("Autoregressive density error")
    ax_error.set_ylabel("Node RMSE")
    ax_error.legend()

    truth_mass = metrics["truth_mass"][local_index]
    prediction_mass = metrics["prediction_mass"][local_index]
    ax_mass.plot(times, truth_mass - truth_mass[0], color="0.4", label="Truth")
    ax_mass.plot(
        times,
        prediction_mass - prediction_mass[0],
        color="tab:purple",
        label=model_label,
    )
    ax_mass.axhline(0.0, color="0.25", linewidth=0.8)
    ax_mass.set_title("Area-weighted mass drift")
    ax_mass.set_ylabel(r"$M(t)-M(0)$")
    ax_mass.legend()

    ax_bounds.plot(times, np.min(truth, axis=1), color="0.4", label="Truth min")
    ax_bounds.plot(
        times, np.max(truth, axis=1), color="0.4", linestyle="--", label="Truth max"
    )
    ax_bounds.plot(
        times, np.min(prediction, axis=1), color="tab:blue", label="Prediction min"
    )
    ax_bounds.plot(
        times,
        np.max(prediction, axis=1),
        color="tab:blue",
        linestyle="--",
        label="Prediction max",
    )
    ax_bounds.axhline(0.0, color="tab:red", linewidth=0.8)
    ax_bounds.set_title("Density range")
    ax_bounds.set_ylabel("Cell density")
    ax_bounds.legend(fontsize=8, ncol=2)

    ax_chemo.plot(
        times, metrics["truth_weighted_chemo"][local_index], color="0.4", label="Truth"
    )
    ax_chemo.plot(
        times,
        metrics["prediction_weighted_chemo"][local_index],
        color="tab:green",
        label=model_label,
    )
    ax_chemo.set_title("Migration up chemoattractant")
    ax_chemo.set_ylabel(r"$\langle c\rangle_n$")
    ax_chemo.legend()

    ax_com.plot(
        times, metrics["truth_com_displacement"][local_index], color="0.4", label="Truth"
    )
    ax_com.plot(
        times,
        metrics["prediction_com_displacement"][local_index],
        color="tab:orange",
        label=model_label,
    )
    ax_com.set_title("Center-of-mass displacement")
    ax_com.set_ylabel("Displacement")
    ax_com.legend()

    truth_com = metrics["truth_com"][local_index]
    prediction_com = metrics["prediction_com"][local_index]
    ax_com_path.plot(truth_com[:, 0], truth_com[:, 1], color="0.4", label="Truth")
    ax_com_path.plot(
        prediction_com[:, 0], prediction_com[:, 1], color="tab:orange", label=model_label
    )
    ax_com_path.scatter(
        [truth_com[0, 0]], [truth_com[0, 1]], marker="o", color="tab:green", label="Start"
    )
    ax_com_path.set_title("Center-of-mass path")
    ax_com_path.set_xlabel("x")
    ax_com_path.set_ylabel("y")
    all_x = np.concatenate((truth_com[:, 0], prediction_com[:, 0]))
    all_y = np.concatenate((truth_com[:, 1], prediction_com[:, 1]))
    x_span = max(float(np.ptp(all_x)), 1.0e-4)
    y_span = max(float(np.ptp(all_y)), 1.0e-4)
    ax_com_path.set_xlim(float(np.min(all_x)) - 0.08 * x_span, float(np.max(all_x)) + 0.08 * x_span)
    ax_com_path.set_ylim(float(np.min(all_y)) - 0.08 * y_span, float(np.max(all_y)) + 0.08 * y_span)
    ax_com_path.legend(fontsize=8)

    for axis in axes.ravel()[:-1]:
        axis.set_xlabel("Time")
        axis.grid(alpha=0.25)
    ax_com_path.grid(alpha=0.25)
    figure.suptitle(f"Held-out trajectory {trajectory}: rollout diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_test_summary(
    trajectory_ids: np.ndarray,
    metrics: Dict[str, np.ndarray],
    selected: Sequence[int],
    output_path: Path,
    *,
    dt: float,
    model_label: str,
    dpi: int,
) -> None:
    """Plot aggregate held-out rollout performance and persistence comparison."""
    times = np.arange(metrics["model_rmse_by_time"].shape[1]) * dt
    model_time = metrics["model_rmse_by_time"]
    persistence_time = metrics["persistence_rmse_by_time"]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    ax_time, ax_baseline, ax_mass, ax_rank = axes.ravel()

    ax_time.plot(times, np.mean(model_time, axis=0), label=model_label)
    ax_time.fill_between(
        times,
        np.quantile(model_time, 0.25, axis=0),
        np.quantile(model_time, 0.75, axis=0),
        alpha=0.2,
        label=f"{model_label} IQR",
    )
    ax_time.plot(
        times, np.mean(persistence_time, axis=0), color="0.45", label="Persistence"
    )
    ax_time.set_title("Held-out autoregressive error")
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Mean node RMSE")
    ax_time.legend(fontsize=8)

    model_rmse = metrics["model_rollout_rmse"]
    persistence_rmse = metrics["persistence_rollout_rmse"]
    upper = max(float(np.max(model_rmse)), float(np.max(persistence_rmse)), 1.0e-12)
    ax_baseline.scatter(persistence_rmse, model_rmse, alpha=0.8)
    ax_baseline.plot([0.0, upper], [0.0, upper], color="0.35", linestyle="--")
    ax_baseline.set_xlim(0.0, upper * 1.04)
    ax_baseline.set_ylim(0.0, upper * 1.04)
    ax_baseline.set_title("Comparison with persistence")
    ax_baseline.set_xlabel("Persistence rollout RMSE")
    ax_baseline.set_ylabel(f"{model_label} rollout RMSE")

    absolute_mass_error = np.abs(metrics["mass_error"])
    ax_mass.plot(times, np.median(absolute_mass_error, axis=0), label="Median")
    ax_mass.fill_between(
        times,
        np.quantile(absolute_mass_error, 0.25, axis=0),
        np.quantile(absolute_mass_error, 0.75, axis=0),
        alpha=0.2,
        label="IQR",
    )
    ax_mass.plot(times, np.max(absolute_mass_error, axis=0), color="tab:red", label="Maximum")
    ax_mass.set_title("Predicted total-mass error")
    ax_mass.set_xlabel("Time")
    ax_mass.set_ylabel(r"$|\hat M(t)-M(t)|$")
    ax_mass.legend(fontsize=8)

    order = np.argsort(model_rmse)
    ordered_ids = trajectory_ids[order]
    selected_set = set(int(value) for value in selected)
    colors = ["tab:orange" if int(value) in selected_set else "tab:blue" for value in ordered_ids]
    ax_rank.bar(np.arange(order.size), model_rmse[order], color=colors, alpha=0.82)
    ax_rank.set_title("Test trajectories ranked by rollout RMSE")
    ax_rank.set_xlabel("Rank")
    ax_rank.set_ylabel("Rollout RMSE")

    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.suptitle(f"{model_label} held-out chemotaxis rollouts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(
    path: Path, trajectory_ids: np.ndarray, metrics: Dict[str, np.ndarray]
) -> None:
    """Write one comparison row per held-out trajectory."""
    fields = (
        "trajectory_id",
        "model_rollout_rmse",
        "persistence_rollout_rmse",
        "rmse_improvement_fraction",
        "final_step_rmse",
        "max_abs_mass_error",
        "negative_density_fraction",
        "final_weighted_chemo_error",
        "final_com_displacement_error",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, trajectory in enumerate(trajectory_ids):
            model_rmse = float(metrics["model_rollout_rmse"][index])
            persistence_rmse = float(metrics["persistence_rollout_rmse"][index])
            improvement = (
                1.0 - model_rmse / persistence_rmse
                if persistence_rmse > 0.0
                else float("nan")
            )
            writer.writerow(
                {
                    "trajectory_id": int(trajectory),
                    "model_rollout_rmse": model_rmse,
                    "persistence_rollout_rmse": persistence_rmse,
                    "rmse_improvement_fraction": improvement,
                    "final_step_rmse": float(metrics["final_step_rmse"][index]),
                    "max_abs_mass_error": float(
                        np.max(np.abs(metrics["mass_error"][index]))
                    ),
                    "negative_density_fraction": float(
                        np.mean(metrics["negative_fraction"][index])
                    ),
                    "final_weighted_chemo_error": float(
                        metrics["prediction_weighted_chemo"][index, -1]
                        - metrics["truth_weighted_chemo"][index, -1]
                    ),
                    "final_com_displacement_error": float(
                        metrics["prediction_com_displacement"][index, -1]
                        - metrics["truth_com_displacement"][index, -1]
                    ),
                }
            )


def summary_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    model_rmse = metrics["model_rollout_rmse"]
    persistence_rmse = metrics["persistence_rollout_rmse"]
    return {
        "mean_model_rollout_rmse": float(np.mean(model_rmse)),
        "median_model_rollout_rmse": float(np.median(model_rmse)),
        "mean_persistence_rollout_rmse": float(np.mean(persistence_rmse)),
        "fraction_beating_persistence": float(np.mean(model_rmse < persistence_rmse)),
        "mean_final_step_rmse": float(np.mean(metrics["final_step_rmse"])),
        "maximum_absolute_mass_error": float(np.max(np.abs(metrics["mass_error"]))),
        "mean_negative_density_fraction": float(np.mean(metrics["negative_fraction"])),
    }


def checkpoint_feature_options(
    checkpoint: Dict[str, object],
) -> Dict[str, bool]:
    """Recover node-feature choices, including legacy-checkpoint defaults."""
    stored = checkpoint.get("feature_options")
    if stored is not None:
        if not isinstance(stored, dict):
            raise ValueError("Checkpoint feature_options must be a dictionary.")
        source = stored
    else:
        training_args = checkpoint.get("training_args", {})
        source = training_args if isinstance(training_args, dict) else {}
    options = {
        "include_chi": source.get("include_chi", True),
        "include_drift_velocity": source.get("include_drift_velocity", False),
        "include_absolute_positions": source.get(
            "include_absolute_positions", True
        ),
        "include_boundary_distances": source.get(
            "include_boundary_distances", False
        ),
    }
    invalid = [name for name, value in options.items() if not isinstance(value, bool)]
    if invalid:
        raise ValueError(
            "Checkpoint feature option(s) must be boolean: " + ", ".join(invalid)
        )
    return options


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot held-out autoregressive GNN rollouts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Dataset override; defaults to checkpoint data_path.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to CHECKPOINT_DIR/rollout_plots.",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=None,
        help="Defaults to every stored transition.",
    )
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument("--mass-projection", dest="mass_projection", action="store_true")
    projection.add_argument("--no-mass-projection", dest="mass_projection", action="store_false")
    parser.set_defaults(mass_projection=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trajectories", nargs="+", type=int)
    selection.add_argument("--all-test", action="store_true")
    parser.add_argument("--snapshot-steps", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap", default="magma")
    contours = parser.add_mutually_exclusive_group()
    contours.add_argument(
        "--chemo-contours",
        dest="chemo_contours",
        action="store_true",
        help="Overlay chemoattractant contours on truth and prediction panels.",
    )
    contours.add_argument(
        "--no-chemo-contours",
        dest="chemo_contours",
        action="store_false",
        help="Disable chemoattractant contour overlays.",
    )
    parser.set_defaults(chemo_contours=True)
    parser.add_argument(
        "--chemo-contour-levels",
        type=int,
        default=6,
        help="Number of chemoattractant contour levels (default: 6).",
    )
    parser.add_argument(
        "--chemo-contour-alpha",
        type=float,
        default=0.45,
        help="Chemoattractant contour opacity in [0, 1] (default: 0.45).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args.batch_size <= 0 or args.dpi <= 0:
        raise ValueError("--batch-size and --dpi must be positive.")
    if args.chemo_contour_levels <= 0:
        raise ValueError("--chemo-contour-levels must be positive.")
    if not (0.0 <= args.chemo_contour_alpha <= 1.0):
        raise ValueError("--chemo-contour-alpha must be in [0, 1].")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    data_path = (
        args.data.expanduser().resolve()
        if args.data is not None
        else Path(str(checkpoint["data_path"])).expanduser().resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "rollout_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_archive(data_path)
    states = np.asarray(data["rollout_states"], dtype=np.float32)
    num_steps = states.shape[1] - 1 if args.rollout_steps is None else args.rollout_steps
    if num_steps > states.shape[1] - 1:
        raise ValueError(
            f"--rollout-steps cannot exceed {states.shape[1] - 1}."
        )
    if not np.isclose(float(checkpoint["dt"]), float(np.asarray(data["dt"]).item())):
        raise ValueError("Checkpoint dt does not match dataset dt.")

    splits = checkpoint["split_trajectory_ids"]
    if not isinstance(splits, dict) or "test" not in splits:
        raise KeyError("Checkpoint does not contain test trajectory IDs.")
    test_ids = np.asarray(splits["test"], dtype=np.int64)
    if test_ids.size == 0 or test_ids.min() < 0 or test_ids.max() >= states.shape[0]:
        raise ValueError("Checkpoint test trajectory IDs do not match the dataset.")

    normalization = Normalization(**dict(checkpoint["normalization"]))
    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = choose_device(args.device)
    model = model.to(device)
    model.eval()
    target_type = str(checkpoint["target_type"])
    checkpoint_projection = checkpoint.get("mass_projection", False)
    if not isinstance(checkpoint_projection, bool):
        raise ValueError("Checkpoint mass_projection must be boolean.")
    mass_projection = (
        checkpoint_projection if args.mass_projection is None else args.mass_projection
    )
    model_label = str(checkpoint.get("model_name", type(model).__name__))
    feature_options = checkpoint_feature_options(checkpoint)

    truth = states[test_ids, : num_steps + 1]
    prediction = predict_rollouts(
        model,
        data,
        test_ids,
        num_steps=num_steps,
        normalization=normalization,
        target_type=target_type,
        mass_projection=mass_projection,
        device=device,
        batch_size=args.batch_size,
        **feature_options,
    )
    metrics = compute_metrics(
        truth,
        prediction,
        cell_areas=np.asarray(data["cell_areas"], dtype=np.float64)[test_ids],
        chemo=np.asarray(data["chemoattractant"], dtype=np.float64)[test_ids, :, 0],
        centers=np.asarray(data["cell_centers"], dtype=np.float64)[test_ids],
    )
    if args.all_test:
        selected = [int(value) for value in test_ids]
    elif args.trajectories is not None:
        selected = validate_selected(args.trajectories, test_ids)
    else:
        selected = representative_trajectories(test_ids, metrics["model_rollout_rmse"])
    snapshot_steps = validate_snapshot_steps(
        args.snapshot_steps or default_snapshot_steps(num_steps + 1), num_steps + 1
    )
    trajectory_to_local = {int(value): index for index, value in enumerate(test_ids)}
    dt = float(np.asarray(data["dt"]).item())

    prediction_path = output_dir / "test_rollout_predictions.npz"
    np.savez_compressed(
        prediction_path,
        trajectory_ids=test_ids,
        truth=truth,
        prediction=prediction,
        persistence=metrics["persistence"],
        time=np.arange(num_steps + 1, dtype=np.float64) * dt,
        model_rmse_by_time=metrics["model_rmse_by_time"],
        persistence_rmse_by_time=metrics["persistence_rmse_by_time"],
        mass_error=metrics["mass_error"],
    )
    metrics_path = output_dir / "test_rollout_metrics.csv"
    write_metrics_csv(metrics_path, test_ids, metrics)
    summary = summary_metrics(metrics)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "data": str(data_path),
            "model": model_label,
            "target_type": target_type,
            "rollout_steps": int(num_steps),
            "mass_projection": bool(mass_projection),
            "chemo_contours": bool(args.chemo_contours),
            "chemo_contour_levels": int(args.chemo_contour_levels),
            "chemo_contour_alpha": float(args.chemo_contour_alpha),
            **feature_options,
            "test_trajectories": int(test_ids.size),
        }
    )
    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    summary_figure = output_dir / "test_rollout_summary.png"
    plot_test_summary(
        test_ids, metrics, selected, summary_figure,
        dt=dt, model_label=model_label, dpi=args.dpi,
    )

    written = [prediction_path, metrics_path, summary_path, summary_figure]
    for trajectory in selected:
        local = trajectory_to_local[trajectory]
        spatial_path = output_dir / f"trajectory_{trajectory:03d}_spatial_rollout.png"
        diagnostic_path = output_dir / f"trajectory_{trajectory:03d}_diagnostics.png"
        plot_spatial_comparison(
            data,
            truth[local, ..., 0],
            prediction[local, ..., 0],
            trajectory,
            spatial_path,
            snapshot_steps=snapshot_steps,
            dt=dt,
            model_label=model_label,
            dpi=args.dpi,
            cmap=args.cmap,
            chemo_contours=args.chemo_contours,
            chemo_contour_levels=args.chemo_contour_levels,
            chemo_contour_alpha=args.chemo_contour_alpha,
        )
        plot_trajectory_diagnostics(
            truth[local, ..., 0],
            prediction[local, ..., 0],
            metrics,
            local,
            trajectory,
            diagnostic_path,
            dt=dt,
            model_label=model_label,
            dpi=args.dpi,
        )
        written.extend((spatial_path, diagnostic_path))

    print(f"device: {device}")
    print(f"model: {model_label}; target={target_type}; rollout steps={num_steps}")
    print(
        "optional node inputs: "
        f"chi={feature_options['include_chi']}, "
        f"drift_velocity={feature_options['include_drift_velocity']}, "
        f"absolute_positions={feature_options['include_absolute_positions']}, "
        f"boundary_distances={feature_options['include_boundary_distances']}"
    )
    print(f"test trajectories: {test_ids.tolist()}")
    print(f"plotted trajectories: {selected}")
    print(json.dumps(summary_metrics(metrics), indent=2, sort_keys=True))
    print("outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
