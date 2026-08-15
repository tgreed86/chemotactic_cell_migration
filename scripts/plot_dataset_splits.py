#!/usr/bin/env python3
"""Compare chemotaxis train, validation, and test trajectory splits.

The script reads the exact trajectory IDs stored in a training checkpoint and
computes one row of interpretable statistics per trajectory. It compares:

* sampled physical parameters and mixture complexity;
* initial cell-density and chemoattractant fields;
* chemoattractant-source and density-blob spatial coverage;
* the magnitude of the true temporal dynamics; and
* irregular-mesh quality and scale statistics.

The output directory contains PNG figures, a multipage PDF, per-trajectory and
aggregate CSV tables, and a compact JSON summary of the largest measured split
shifts.

Example:

    python scripts/plot_dataset_splits.py \
        --checkpoint runs/sageconv/predict_delta/100ep/checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch to read checkpoint.pt. Use the same "
        "environment used by scripts/train.py."
    ) from exc

from train import load_archive


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test",
}
SPLIT_COLORS = {
    "train": "tab:blue",
    "validation": "tab:orange",
    "test": "tab:green",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    group: str
    discrete: bool = False


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("chi", r"Chemotactic sensitivity $\chi$", "initial"),
    MetricSpec("chemo_source_count", "Chemoattractant sources", "initial", True),
    MetricSpec("density_blob_count", "Initial density blobs", "initial", True),
    MetricSpec("initial_mass", "Initial total cell mass", "initial"),
    MetricSpec("initial_density_peak", "Initial peak density", "initial"),
    MetricSpec("initial_density_std", "Initial spatial density std.", "initial"),
    MetricSpec("chemo_mean", "Spatial mean chemoattractant", "initial"),
    MetricSpec("chemo_std", "Spatial chemoattractant std.", "initial"),
    MetricSpec("chemo_peak", "Peak chemoattractant", "initial"),
    MetricSpec(
        "initial_weighted_chemo",
        r"Initial density-weighted chemo $\langle c\rangle_n$",
        "initial",
    ),
    MetricSpec(
        "initial_com_source_distance",
        "Initial COM to strongest source",
        "initial",
    ),
    MetricSpec("mean_density_blob_sigma", "Mean density-blob width", "initial"),
    MetricSpec("one_step_change_rms", "One-step density-change RMS", "dynamics"),
    MetricSpec("one_step_change_max", "Maximum one-step |change|", "dynamics"),
    MetricSpec(
        "weighted_chemo_gain",
        r"Final gain in $\langle c\rangle_n$",
        "dynamics",
    ),
    MetricSpec("com_displacement", "Final center-of-mass displacement", "dynamics"),
    MetricSpec("final_peak_ratio", "Final / initial peak density", "dynamics"),
    MetricSpec("rollout_peak_density", "Peak density over full rollout", "dynamics"),
    MetricSpec("cell_area_cv", "Cell-area coefficient of variation", "dynamics"),
    MetricSpec("cell_area_min", "Minimum triangle area", "dynamics"),
    MetricSpec("face_length_mean", "Mean shared-face length", "dynamics"),
    MetricSpec("transmissibility_p95", "95th percentile transmissibility", "dynamics"),
)


def load_checkpoint(path: Path) -> Dict[str, object]:
    """Load a trusted training checkpoint and validate its stored split IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")
    split_ids = checkpoint.get("split_trajectory_ids")
    if not isinstance(split_ids, dict):
        raise KeyError("Checkpoint does not contain split_trajectory_ids.")
    missing = [name for name in SPLIT_NAMES if name not in split_ids]
    if missing:
        raise KeyError(f"Checkpoint is missing trajectory split(s): {missing}")
    return checkpoint


def validate_split_ids(
    checkpoint: Mapping[str, object], num_trajectories: int
) -> Dict[str, np.ndarray]:
    """Return disjoint, in-range trajectory IDs in the canonical split order."""
    stored = checkpoint["split_trajectory_ids"]
    if not isinstance(stored, Mapping):
        raise ValueError("split_trajectory_ids must be a mapping.")
    split_ids: Dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for name in SPLIT_NAMES:
        ids = np.asarray(stored[name], dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError(f"Split {name!r} must contain a nonempty ID list.")
        if np.unique(ids).size != ids.size:
            raise ValueError(f"Split {name!r} contains duplicate trajectory IDs.")
        if int(ids.min()) < 0 or int(ids.max()) >= num_trajectories:
            raise ValueError(f"Split {name!r} contains an out-of-range ID.")
        overlap = seen.intersection(int(value) for value in ids)
        if overlap:
            raise ValueError(
                f"Trajectory IDs occur in multiple splits: {sorted(overlap)}"
            )
        seen.update(int(value) for value in ids)
        split_ids[name] = np.sort(ids)
    if len(seen) != num_trajectories:
        omitted = sorted(set(range(num_trajectories)) - seen)
        raise ValueError(
            "Checkpoint splits do not cover every dataset trajectory; omitted IDs: "
            + ", ".join(map(str, omitted))
        )
    return split_ids


def _weighted_mean(values: np.ndarray, areas: np.ndarray) -> float:
    return float(np.sum(values * areas) / np.sum(areas))


def _weighted_std(values: np.ndarray, areas: np.ndarray) -> float:
    mean = _weighted_mean(values, areas)
    return float(np.sqrt(np.sum(areas * np.square(values - mean)) / np.sum(areas)))


def _density_weighted_chemo(
    density: np.ndarray, chemo: np.ndarray, areas: np.ndarray
) -> float:
    weights = density * areas
    return float(np.sum(weights * chemo) / max(float(np.sum(weights)), 1.0e-15))


def _density_center_of_mass(
    density: np.ndarray, centers: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    weights = density * areas
    return np.sum(weights[:, None] * centers, axis=0) / max(
        float(np.sum(weights)), 1.0e-15
    )


def require_fields(data: Mapping[str, np.ndarray], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise KeyError("Dataset is missing diagnostic field(s): " + ", ".join(missing))


def compute_trajectory_statistics(
    data: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute all scalar comparison metrics for every trajectory."""
    require_fields(
        data,
        (
            "rollout_states",
            "trajectory_chi",
            "chemoattractant",
            "cell_areas",
            "cell_centers",
            "shared_face_lengths",
            "transmissibility",
            "chemo_source_count",
            "chemo_source_centers",
            "chemo_source_amplitudes",
            "density_blob_count",
            "density_blob_sigmas",
        ),
    )
    states = np.asarray(data["rollout_states"], dtype=np.float64)[..., 0]
    chemo = np.asarray(data["chemoattractant"], dtype=np.float64)[..., 0]
    areas = np.asarray(data["cell_areas"], dtype=np.float64)
    centers = np.asarray(data["cell_centers"], dtype=np.float64)
    num_trajectories = states.shape[0]
    statistics = {
        metric.key: np.empty(num_trajectories, dtype=np.float64)
        for metric in METRICS
    }
    statistics["chi"][:] = np.asarray(data["trajectory_chi"], dtype=np.float64)
    statistics["chemo_source_count"][:] = np.asarray(
        data["chemo_source_count"], dtype=np.float64
    )
    statistics["density_blob_count"][:] = np.asarray(
        data["density_blob_count"], dtype=np.float64
    )

    for trajectory in range(num_trajectories):
        density0 = states[trajectory, 0]
        density_final = states[trajectory, -1]
        chemo_values = chemo[trajectory]
        area_values = areas[trajectory]
        center_values = centers[trajectory]
        initial_mass = float(np.sum(density0 * area_values))
        initial_com = _density_center_of_mass(
            density0, center_values, area_values
        )
        final_com = _density_center_of_mass(
            density_final, center_values, area_values
        )
        source_count = int(data["chemo_source_count"][trajectory])
        amplitudes = np.asarray(
            data["chemo_source_amplitudes"][trajectory, :source_count],
            dtype=np.float64,
        )
        source_centers = np.asarray(
            data["chemo_source_centers"][trajectory, :source_count],
            dtype=np.float64,
        )
        strongest_source = source_centers[int(np.argmax(amplitudes))]
        blob_count = int(data["density_blob_count"][trajectory])
        blob_sigmas = np.asarray(
            data["density_blob_sigmas"][trajectory, :blob_count],
            dtype=np.float64,
        )
        changes = np.diff(states[trajectory], axis=0)
        cell_area_mean = float(np.mean(area_values))

        statistics["initial_mass"][trajectory] = initial_mass
        statistics["initial_density_peak"][trajectory] = float(np.max(density0))
        statistics["initial_density_std"][trajectory] = _weighted_std(
            density0, area_values
        )
        statistics["chemo_mean"][trajectory] = _weighted_mean(
            chemo_values, area_values
        )
        statistics["chemo_std"][trajectory] = _weighted_std(
            chemo_values, area_values
        )
        statistics["chemo_peak"][trajectory] = float(np.max(chemo_values))
        statistics["initial_weighted_chemo"][trajectory] = (
            _density_weighted_chemo(density0, chemo_values, area_values)
        )
        statistics["initial_com_source_distance"][trajectory] = float(
            np.linalg.norm(initial_com - strongest_source)
        )
        statistics["mean_density_blob_sigma"][trajectory] = float(
            np.mean(blob_sigmas)
        )
        statistics["one_step_change_rms"][trajectory] = float(
            np.sqrt(np.mean(np.square(changes)))
        )
        statistics["one_step_change_max"][trajectory] = float(
            np.max(np.abs(changes))
        )
        statistics["weighted_chemo_gain"][trajectory] = (
            _density_weighted_chemo(
                density_final, chemo_values, area_values
            )
            - statistics["initial_weighted_chemo"][trajectory]
        )
        statistics["com_displacement"][trajectory] = float(
            np.linalg.norm(final_com - initial_com)
        )
        statistics["final_peak_ratio"][trajectory] = float(
            np.max(density_final) / max(float(np.max(density0)), 1.0e-15)
        )
        statistics["rollout_peak_density"][trajectory] = float(
            np.max(states[trajectory])
        )
        statistics["cell_area_cv"][trajectory] = float(
            np.std(area_values) / max(cell_area_mean, 1.0e-15)
        )
        statistics["cell_area_min"][trajectory] = float(np.min(area_values))
        statistics["face_length_mean"][trajectory] = float(
            np.mean(data["shared_face_lengths"][trajectory])
        )
        statistics["transmissibility_p95"][trajectory] = float(
            np.quantile(data["transmissibility"][trajectory], 0.95)
        )

    for key, values in statistics.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Computed statistic {key!r} contains non-finite values.")
    return statistics


def empirical_ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov distance without SciPy."""
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    combined = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, combined, side="right") / first.size
    second_cdf = np.searchsorted(second, combined, side="right") / second.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def standardized_mean_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Return (second mean - first mean) divided by pooled standard deviation."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    pooled = np.sqrt(0.5 * (np.var(first, ddof=1) + np.var(second, ddof=1)))
    difference = float(np.mean(second) - np.mean(first))
    if pooled <= 1.0e-15:
        return 0.0 if abs(difference) <= 1.0e-15 else float("nan")
    return difference / float(pooled)


def log2_std_ratio(first: np.ndarray, second: np.ndarray) -> float:
    first_std = float(np.std(first, ddof=1))
    second_std = float(np.std(second, ddof=1))
    if first_std <= 1.0e-15 or second_std <= 1.0e-15:
        return 0.0 if abs(first_std - second_std) <= 1.0e-15 else float("nan")
    return float(np.log2(second_std / first_std))


def compute_split_comparisons(
    statistics: Mapping[str, np.ndarray], split_ids: Mapping[str, np.ndarray]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare validation and test independently against training."""
    comparisons: Dict[str, Dict[str, np.ndarray]] = {}
    train_ids = split_ids["train"]
    for other_name in ("validation", "test"):
        other_ids = split_ids[other_name]
        smd = np.empty(len(METRICS), dtype=np.float64)
        log_std = np.empty(len(METRICS), dtype=np.float64)
        ks = np.empty(len(METRICS), dtype=np.float64)
        for index, metric in enumerate(METRICS):
            train_values = statistics[metric.key][train_ids]
            other_values = statistics[metric.key][other_ids]
            smd[index] = standardized_mean_difference(train_values, other_values)
            log_std[index] = log2_std_ratio(train_values, other_values)
            ks[index] = empirical_ks_distance(train_values, other_values)
        comparisons[other_name] = {
            "standardized_mean_difference": smd,
            "log2_std_ratio": log_std,
            "ks_distance": ks,
        }
    return comparisons


def _boxplot_metric(
    axis: plt.Axes,
    statistics: Mapping[str, np.ndarray],
    split_ids: Mapping[str, np.ndarray],
    metric: MetricSpec,
    *,
    rng: np.random.Generator,
) -> None:
    values = [statistics[metric.key][split_ids[name]] for name in SPLIT_NAMES]
    boxes = axis.boxplot(
        values,
        positions=np.arange(3),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.2},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
    )
    for patch, name in zip(boxes["boxes"], SPLIT_NAMES):
        patch.set_facecolor(SPLIT_COLORS[name])
        patch.set_alpha(0.22)
        patch.set_edgecolor(SPLIT_COLORS[name])
    for position, name, split_values in zip(np.arange(3), SPLIT_NAMES, values):
        jitter = rng.uniform(-0.16, 0.16, size=split_values.size)
        axis.scatter(
            position + jitter,
            split_values,
            s=15,
            alpha=0.58,
            color=SPLIT_COLORS[name],
            edgecolors="none",
        )
        axis.scatter(
            [position],
            [np.mean(split_values)],
            marker="D",
            s=28,
            color=SPLIT_COLORS[name],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axis.set_xticks(np.arange(3), [SPLIT_LABELS[name] for name in SPLIT_NAMES])
    axis.set_title(metric.label)
    axis.grid(axis="y", alpha=0.24)
    if metric.discrete:
        lower = int(np.floor(min(float(np.min(item)) for item in values)))
        upper = int(np.ceil(max(float(np.max(item)) for item in values)))
        axis.set_yticks(np.arange(lower, upper + 1))


def plot_metric_group(
    statistics: Mapping[str, np.ndarray],
    split_ids: Mapping[str, np.ndarray],
    *,
    group: str,
    title: str,
    seed: int,
) -> plt.Figure:
    metrics = [metric for metric in METRICS if metric.group == group]
    columns = 3 if len(metrics) <= 12 else 4
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.55 * columns, 3.35 * rows), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(seed)
    for axis, metric in zip(axes_array, metrics):
        _boxplot_metric(axis, statistics, split_ids, metric, rng=rng)
    for axis in axes_array[len(metrics) :]:
        axis.set_visible(False)
    counts = ", ".join(
        f"{SPLIT_LABELS[name]} n={split_ids[name].size}" for name in SPLIT_NAMES
    )
    figure.suptitle(f"{title}\n{counts}; diamonds show means")
    return figure


def plot_spatial_coverage(
    data: Mapping[str, np.ndarray], split_ids: Mapping[str, np.ndarray]
) -> plt.Figure:
    """Plot sampled chemo-source and density-blob centers by split."""
    require_fields(
        data,
        (
            "chemo_source_count",
            "chemo_source_centers",
            "density_blob_count",
            "density_blob_centers",
            "lx",
            "ly",
        ),
    )
    lx = float(np.asarray(data["lx"]).item())
    ly = float(np.asarray(data["ly"]).item())
    figure, axes = plt.subplots(
        2, 3, figsize=(12.0, 7.8), constrained_layout=True, sharex=True, sharey=True
    )
    rows = (
        ("chemo_source_count", "chemo_source_centers", "Chemo sources"),
        ("density_blob_count", "density_blob_centers", "Initial density blobs"),
    )
    for row, (count_key, center_key, row_label) in enumerate(rows):
        for column, split_name in enumerate(SPLIT_NAMES):
            axis = axes[row, column]
            collected = []
            for trajectory in split_ids[split_name]:
                count = int(data[count_key][trajectory])
                collected.append(
                    np.asarray(data[center_key][trajectory, :count], dtype=np.float64)
                )
            points = np.concatenate(collected, axis=0)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                color=SPLIT_COLORS[split_name],
                s=24,
                alpha=0.62,
                edgecolor="white",
                linewidth=0.35,
            )
            axis.set_xlim(0.0, lx)
            axis.set_ylim(0.0, ly)
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{SPLIT_LABELS[split_name]}: {points.shape[0]} {row_label.lower()}"
            )
            if column == 0:
                axis.set_ylabel(f"{row_label}\ny")
            if row == 1:
                axis.set_xlabel("x")
    figure.suptitle("Spatial coverage of sampled initial conditions")
    return figure


def _finite_limit(values: np.ndarray, minimum: float) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(minimum, float(np.max(finite)) if finite.size else minimum)


def plot_shift_heatmap(
    comparisons: Mapping[str, Mapping[str, np.ndarray]],
) -> plt.Figure:
    """Summarize changes in means, spreads, and empirical distributions."""
    labels = [metric.label for metric in METRICS]
    columns = ["Validation − Train", "Test − Train"]
    smd = np.column_stack(
        [comparisons[name]["standardized_mean_difference"] for name in ("validation", "test")]
    )
    log_std = np.column_stack(
        [comparisons[name]["log2_std_ratio"] for name in ("validation", "test")]
    )
    ks = np.column_stack(
        [comparisons[name]["ks_distance"] for name in ("validation", "test")]
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 11.0), constrained_layout=True, sharey=True
    )
    panels = (
        (
            smd,
            "Standardized mean difference",
            "coolwarm",
            -_finite_limit(smd, 1.0),
            _finite_limit(smd, 1.0),
            ".2f",
        ),
        (
            log_std,
            r"Spread change: $\log_2(\sigma/\sigma_{train})$",
            "PiYG",
            -_finite_limit(log_std, 1.0),
            _finite_limit(log_std, 1.0),
            ".2f",
        ),
        (ks, "Empirical KS distance", "viridis", 0.0, 1.0, ".2f"),
    )
    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        values, title, cmap, lower, upper, number_format = panel
        image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=lower, vmax=upper)
        axis.set_xticks(np.arange(2), columns, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(labels)))
        if panel_index == 0:
            axis.set_yticklabels(labels)
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                text = "n/a" if not np.isfinite(value) else format(value, number_format)
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="white" if abs(value) > 0.62 * max(abs(lower), abs(upper)) else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    figure.suptitle(
        "How validation and test differ from training\n"
        "Positive mean shifts indicate larger validation/test values; KS measures any distributional change"
    )
    return figure


def write_trajectory_csv(
    path: Path,
    statistics: Mapping[str, np.ndarray],
    split_ids: Mapping[str, np.ndarray],
) -> None:
    split_by_trajectory = {
        int(trajectory): split_name
        for split_name, ids in split_ids.items()
        for trajectory in ids
    }
    fields = ["trajectory_id", "split", *[metric.key for metric in METRICS]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trajectory in sorted(split_by_trajectory):
            row: Dict[str, object] = {
                "trajectory_id": trajectory,
                "split": split_by_trajectory[trajectory],
            }
            row.update(
                {metric.key: float(statistics[metric.key][trajectory]) for metric in METRICS}
            )
            writer.writerow(row)


def write_split_summary_csv(
    path: Path,
    statistics: Mapping[str, np.ndarray],
    split_ids: Mapping[str, np.ndarray],
) -> None:
    fields = (
        "split",
        "trajectory_count",
        "metric",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "min",
        "max",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split_name in SPLIT_NAMES:
            ids = split_ids[split_name]
            for metric in METRICS:
                values = statistics[metric.key][ids]
                writer.writerow(
                    {
                        "split": split_name,
                        "trajectory_count": int(ids.size),
                        "metric": metric.key,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )


def write_comparison_csv(
    path: Path, comparisons: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    fields = (
        "comparison",
        "metric",
        "standardized_mean_difference",
        "log2_std_ratio",
        "ks_distance",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for other_name in ("validation", "test"):
            values = comparisons[other_name]
            for index, metric in enumerate(METRICS):
                writer.writerow(
                    {
                        "comparison": f"{other_name}_vs_train",
                        "metric": metric.key,
                        "standardized_mean_difference": float(
                            values["standardized_mean_difference"][index]
                        ),
                        "log2_std_ratio": float(values["log2_std_ratio"][index]),
                        "ks_distance": float(values["ks_distance"][index]),
                    }
                )


def build_json_summary(
    checkpoint_path: Path,
    data_path: Path,
    split_ids: Mapping[str, np.ndarray],
    comparisons: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "split_counts": {name: int(split_ids[name].size) for name in SPLIT_NAMES},
        "largest_shifts": {},
    }
    largest: Dict[str, object] = {}
    for other_name in ("validation", "test"):
        comparison = comparisons[other_name]
        smd = comparison["standardized_mean_difference"]
        ks = comparison["ks_distance"]
        smd_order = np.argsort(-np.nan_to_num(np.abs(smd), nan=-1.0))[:5]
        ks_order = np.argsort(-ks)[:5]
        largest[other_name] = {
            "by_absolute_standardized_mean_difference": [
                {
                    "metric": METRICS[index].key,
                    "value": float(smd[index]),
                }
                for index in smd_order
            ],
            "by_ks_distance": [
                {
                    "metric": METRICS[index].key,
                    "value": float(ks[index]),
                }
                for index in ks_order
            ],
        }
    summary["largest_shifts"] = largest
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot statistical differences among checkpoint trajectory splits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset override; defaults to checkpoint data_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to CHECKPOINT_DIR/split_diagnostics.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1234, help="Plot-jitter seed.")
    parser.add_argument(
        "--skip-pdf", action="store_true", help="Do not create the multipage PDF."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    data_path = (
        args.data.expanduser().resolve()
        if args.data is not None
        else Path(str(checkpoint.get("data_path", ""))).expanduser().resolve()
    )
    if not str(checkpoint.get("data_path", "")) and args.data is None:
        raise KeyError("Checkpoint has no data_path; supply --data explicitly.")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "split_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_archive(data_path)
    split_ids = validate_split_ids(checkpoint, data["rollout_states"].shape[0])
    statistics = compute_trajectory_statistics(data)
    comparisons = compute_split_comparisons(statistics, split_ids)

    figures = (
        (
            "initial_condition_distributions.png",
            plot_metric_group(
                statistics,
                split_ids,
                group="initial",
                title="Physical parameters and initial-condition distributions",
                seed=args.seed,
            ),
        ),
        (
            "dynamics_and_mesh_distributions.png",
            plot_metric_group(
                statistics,
                split_ids,
                group="dynamics",
                title="True dynamics and mesh distributions",
                seed=args.seed + 1,
            ),
        ),
        ("spatial_coverage.png", plot_spatial_coverage(data, split_ids)),
        ("split_shift_heatmap.png", plot_shift_heatmap(comparisons)),
    )
    written: List[Path] = []
    for filename, figure in figures:
        path = output_dir / filename
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        written.append(path)
    if not args.skip_pdf:
        pdf_path = output_dir / "split_diagnostics.pdf"
        with PdfPages(pdf_path) as pdf:
            for _, figure in figures:
                pdf.savefig(figure, bbox_inches="tight")
        written.append(pdf_path)
    for _, figure in figures:
        plt.close(figure)

    trajectory_csv = output_dir / "trajectory_statistics.csv"
    split_csv = output_dir / "split_summary_statistics.csv"
    comparison_csv = output_dir / "split_comparisons.csv"
    summary_path = output_dir / "split_diagnostics_summary.json"
    write_trajectory_csv(trajectory_csv, statistics, split_ids)
    write_split_summary_csv(split_csv, statistics, split_ids)
    write_comparison_csv(comparison_csv, comparisons)
    summary = build_json_summary(
        checkpoint_path, data_path, split_ids, comparisons
    )
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    written.extend((trajectory_csv, split_csv, comparison_csv, summary_path))

    print(
        "split trajectories: "
        + ", ".join(f"{name}={split_ids[name].size}" for name in SPLIT_NAMES)
    )
    for other_name in ("validation", "test"):
        smd = comparisons[other_name]["standardized_mean_difference"]
        largest = int(np.nanargmax(np.abs(smd)))
        print(
            f"largest {other_name}-vs-train mean shift: "
            f"{METRICS[largest].key} (SMD={smd[largest]:.3f})"
        )
    print("outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
