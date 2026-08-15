#!/usr/bin/env python3
"""Visualize chemotactic cell-migration trajectory archives.

The script understands the trajectory-specific mesh format produced by
``generate_chemotactic_cell_migration_data.py``. It creates:

* a dataset-level diagnostic summary as PNG;
* one spatial/dynamical summary PNG per selected trajectory;
* one density-evolution GIF per selected trajectory; and
* a PDF copy of the dataset-level diagnostic summary.

Example
-------

    /Applications/anaconda3/bin/python \
        scripts/visualize_chemotactic_cell_migration_data.py \
        --data data/rollout/chemotaxis_train.npz
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Matplotlib otherwise tries to create a cache under the user's home directory.
_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.figure import Figure
from PIL import Image


REQUIRED_ARRAYS = (
    "rollout_states",
    "chemoattractant",
    "drift_velocity",
    "mesh_vertices",
    "triangles",
    "cell_centers",
    "cell_areas",
    "undirected_edge_index",
    "center_distances",
    "shared_face_midpoints",
    "shared_face_normals",
    "total_mass",
    "dt",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create static and animated diagnostics for chemotaxis data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/rollout/chemotaxis_train.npz"),
        help="Input trajectory archive.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/chemotaxis_visualization"),
        help="Directory for PNG and GIF files.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=Path("output/pdf/chemotaxis_dataset_diagnostics.pdf"),
        help="Dataset-level diagnostic PDF path.",
    )
    parser.add_argument(
        "--trajectories",
        default="auto",
        help=(
            "Comma-separated trajectory indices, 'all', or 'auto'. Auto shows every "
            "trajectory when there are at most four and otherwise chooses four "
            "evenly spaced examples."
        ),
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=65,
        help="Maximum number of evenly spaced frames in each GIF.",
    )
    parser.add_argument(
        "--gif-fps", type=int, default=12, help="GIF playback frames per second."
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    parser.add_argument(
        "--skip-gif", action="store_true", help="Create only static outputs."
    )
    parser.add_argument(
        "--skip-pdf", action="store_true", help="Do not create the PDF report."
    )
    return parser.parse_args(argv)


def load_archive(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive.files]
        if missing:
            raise KeyError(f"Dataset is missing required arrays: {missing}")
        data = {name: archive[name] for name in archive.files}

    rollout = data["rollout_states"]
    if rollout.ndim != 4 or rollout.shape[-1] != 1:
        raise ValueError(
            "rollout_states must have shape [trajectories, times, cells, 1]."
        )
    num_samples, _, num_cells, _ = rollout.shape
    expected_shapes = {
        "chemoattractant": (num_samples, num_cells, 1),
        "drift_velocity": (num_samples, num_cells, 2),
        "cell_centers": (num_samples, num_cells, 2),
        "cell_areas": (num_samples, num_cells),
    }
    for name, expected in expected_shapes.items():
        if data[name].shape != expected:
            raise ValueError(
                f"{name} has shape {data[name].shape}, expected {expected}."
            )
    if not all(np.isfinite(data[name]).all() for name in REQUIRED_ARRAYS):
        raise ValueError("The archive contains non-finite values.")
    return data


def parse_trajectory_selection(spec: str, num_samples: int) -> List[int]:
    selection = spec.strip().lower()
    if selection == "all":
        return list(range(num_samples))
    if selection == "auto":
        if num_samples <= 4:
            return list(range(num_samples))
        return np.linspace(0, num_samples - 1, 4, dtype=np.int64).tolist()
    result: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if not 0 <= index < num_samples:
            raise ValueError(
                f"Trajectory {index} is outside the valid range [0, {num_samples - 1}]."
            )
        if index not in result:
            result.append(index)
    if not result:
        raise ValueError("No trajectories were selected.")
    return result


def trajectory_diagnostics(
    data: Dict[str, np.ndarray], sample: int
) -> Dict[str, np.ndarray]:
    density = data["rollout_states"][sample, ..., 0].astype(np.float64)
    areas = data["cell_areas"][sample].astype(np.float64)
    centers = data["cell_centers"][sample].astype(np.float64)
    chemo = data["chemoattractant"][sample, :, 0].astype(np.float64)
    edge_index = data["undirected_edge_index"][sample]
    edge_distance = data["center_distances"][sample].astype(np.float64)
    dt = float(data["dt"])

    mass = np.sum(density * areas[None, :], axis=1)
    center_of_mass = np.einsum("tc,c,cd->td", density, areas, centers) / mass[:, None]
    weighted_chemo = np.sum(density * areas[None, :] * chemo[None, :], axis=1) / mass
    cell_i, cell_j = edge_index
    max_edge_gradient = np.max(
        np.abs(density[:, cell_j] - density[:, cell_i]) / edge_distance[None, :],
        axis=1,
    )
    return {
        "time": np.arange(density.shape[0], dtype=np.float64) * dt,
        "density": density,
        "mass": mass,
        "mass_error": mass - mass[0],
        "center_of_mass": center_of_mass,
        "center_of_mass_displacement": np.linalg.norm(
            center_of_mass - center_of_mass[0], axis=1
        ),
        "weighted_chemo": weighted_chemo,
        "peak_density": np.max(density, axis=1),
        "max_edge_gradient": max_edge_gradient,
    }


def face_peclet_numbers(data: Dict[str, np.ndarray], sample: int) -> np.ndarray:
    """Evaluate the face-normal chemotactic Péclet number from source metadata."""
    count = int(data["chemo_source_count"][sample])
    points = data["shared_face_midpoints"][sample].astype(np.float64)
    normals = data["shared_face_normals"][sample].astype(np.float64)
    centers = data["chemo_source_centers"][sample, :count].astype(np.float64)
    amplitudes = data["chemo_source_amplitudes"][sample, :count].astype(np.float64)
    sigmas = data["chemo_source_sigmas"][sample, :count].astype(np.float64)
    delta = points[:, None, :] - centers[None, :, :]
    sigma2 = np.square(sigmas)
    components = amplitudes[None, :] * np.exp(
        -0.5 * np.sum(np.square(delta), axis=2) / sigma2[None, :]
    )
    gradient = np.sum(-components[..., None] * delta / sigma2[None, :, None], axis=1)
    chi = np.asarray(data["chi"])
    sample_chi = float(chi if chi.ndim == 0 else chi[sample])
    normal_speed = sample_chi * np.sum(gradient * normals, axis=1)
    return (
        np.abs(normal_speed)
        * data["center_normal_distances"][sample]
        / float(data["diffusion"])
    )


def triangulation(data: Dict[str, np.ndarray], sample: int) -> mtri.Triangulation:
    vertices = data["mesh_vertices"][sample]
    triangles = data["triangles"][sample]
    return mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)


def style_spatial_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)


def add_cell_map(
    figure: Figure,
    axis: plt.Axes,
    mesh: mtri.Triangulation,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    mesh_lines: bool = False,
) -> object:
    artist = axis.tripcolor(
        mesh,
        facecolors=values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="0.72" if mesh_lines else "none",
        linewidth=0.25 if mesh_lines else 0.0,
    )
    style_spatial_axis(axis, title)
    figure.colorbar(artist, ax=axis, fraction=0.047, pad=0.035)
    return artist


def make_global_figure(
    data: Dict[str, np.ndarray],
    samples: Iterable[int],
    diagnostics: Dict[int, Dict[str, np.ndarray]],
) -> Figure:
    sample_list = list(samples)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    colors = plt.get_cmap("tab10")(np.arange(len(sample_list)))
    panels = (
        ("mass_error", "Area-weighted mass error", r"$M(t)-M(0)$"),
        ("weighted_chemo", "Density-weighted chemoattractant", r"$\langle c\rangle_n$"),
        ("peak_density", "Peak cell density", r"$\max_i n_i$"),
        (
            "center_of_mass_displacement",
            "Center-of-mass displacement",
            r"$\|\bar{x}(t)-\bar{x}(0)\|_2$",
        ),
        ("max_edge_gradient", "Largest graph-edge gradient", r"$\max |\Delta n|/d$"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat[:5], panels):
        for color, sample in zip(colors, sample_list):
            diag = diagnostics[sample]
            axis.plot(
                diag["time"], diag[key], color=color, label=f"Trajectory {sample}"
            )
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    area_axis = axes.flat[5]
    all_areas = [data["cell_areas"][sample] for sample in sample_list]
    shared_bins = np.linspace(
        min(np.min(values) for values in all_areas),
        max(np.max(values) for values in all_areas),
        28,
    )
    for color, sample, values in zip(colors, sample_list, all_areas):
        area_axis.hist(
            values,
            bins=shared_bins,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=f"Trajectory {sample}",
        )
    area_axis.set_title("Coarse-cell area distributions")
    area_axis.set_xlabel("Triangle area")
    area_axis.set_ylabel("Cell count")
    area_axis.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=min(4, len(labels)),
    )
    final_time = (data["rollout_states"].shape[1] - 1) * float(data["dt"])
    figure.suptitle(
        "Chemotactic cell-migration dataset diagnostics\n"
        f"{len(sample_list)} trajectories shown; final time = {final_time:.4g}",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return figure


def make_trajectory_figure(
    data: Dict[str, np.ndarray], sample: int, diag: Dict[str, np.ndarray],
) -> Figure:
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.6))
    mesh = triangulation(data, sample)
    density = diag["density"]
    chemo = data["chemoattractant"][sample, :, 0]
    velocity = data["drift_velocity"][sample]
    centers = data["cell_centers"][sample]
    time = diag["time"]
    middle = density.shape[0] // 2
    density_min = float(np.min(density))
    density_max = float(np.max(density))

    add_cell_map(
        figure,
        axes[0, 0],
        mesh,
        chemo,
        title="Chemoattractant and drift",
        cmap="viridis",
        mesh_lines=True,
    )
    stride = max(1, centers.shape[0] // 80)
    selected = np.arange(0, centers.shape[0], stride)
    axes[0, 0].quiver(
        centers[selected, 0],
        centers[selected, 1],
        velocity[selected, 0],
        velocity[selected, 1],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.003,
        alpha=0.85,
    )
    for axis, frame, label in (
        (axes[0, 1], 0, "Initial density"),
        (axes[0, 2], middle, "Midpoint density"),
        (axes[0, 3], density.shape[0] - 1, "Final density"),
    ):
        add_cell_map(
            figure,
            axis,
            mesh,
            density[frame],
            title=f"{label}\nt = {time[frame]:.4g}",
            cmap="magma",
            vmin=density_min,
            vmax=density_max,
        )

    change = density[-1] - density[0]
    change_limit = float(np.max(np.abs(change)))
    add_cell_map(
        figure,
        axes[1, 0],
        mesh,
        change,
        title="Final minus initial density",
        cmap="coolwarm",
        vmin=-change_limit,
        vmax=change_limit,
    )
    line_panels = (
        (axes[1, 1], diag["mass_error"], "Mass conservation", r"$M(t)-M(0)$"),
        (
            axes[1, 2],
            diag["weighted_chemo"],
            "Migration up chemoattractant",
            r"$\langle c\rangle_n$",
        ),
        (
            axes[1, 3],
            diag["center_of_mass_displacement"],
            "Center-of-mass motion",
            "Displacement",
        ),
    )
    for axis, values, title, ylabel in line_panels:
        axis.plot(time, values, color=plt.get_cmap("tab10")(sample % 10), linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    peclet = face_peclet_numbers(data, sample)
    source_count = int(data["chemo_source_count"][sample])
    figure.suptitle(
        f"Trajectory {sample}: {density.shape[1]} cells, {source_count} attractors, "
        f"median face Pe = {np.median(peclet):.3g}",
        y=0.97,
    )
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.075, top=0.84, wspace=0.42, hspace=0.38,
    )
    return figure


def make_density_gif(
    data: Dict[str, np.ndarray],
    sample: int,
    diag: Dict[str, np.ndarray],
    path: Path,
    *,
    max_frames: int,
    fps: int,
) -> None:
    if max_frames <= 1:
        raise ValueError("gif_frames must be greater than one.")
    if fps <= 0:
        raise ValueError("gif_fps must be positive.")

    density = diag["density"]
    time = diag["time"]
    chemo = data["chemoattractant"][sample, :, 0]
    velocity = data["drift_velocity"][sample]
    centers = data["cell_centers"][sample]
    mesh = triangulation(data, sample)
    frame_indices = np.unique(
        np.rint(
            np.linspace(0, density.shape[0] - 1, min(max_frames, density.shape[0]))
        ).astype(np.int64)
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    density_artist = add_cell_map(
        figure,
        axes[0],
        mesh,
        density[0],
        title="Cell density",
        cmap="magma",
        vmin=float(np.min(density)),
        vmax=float(np.max(density)),
    )
    add_cell_map(
        figure,
        axes[1],
        mesh,
        chemo,
        title="Static chemoattractant and drift",
        cmap="viridis",
        mesh_lines=True,
    )
    stride = max(1, centers.shape[0] // 90)
    selected = np.arange(0, centers.shape[0], stride)
    axes[1].quiver(
        centers[selected, 0],
        centers[selected, 1],
        velocity[selected, 0],
        velocity[selected, 1],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.003,
        alpha=0.85,
    )

    def update(frame_number: int) -> Sequence[object]:
        frame = int(frame_indices[frame_number])
        density_artist.set_array(density[frame])
        axes[0].set_title(
            "Cell density\n"
            f"t = {time[frame]:.4g}, max n = {diag['peak_density'][frame]:.3g}, "
            f"<c>_n = {diag['weighted_chemo'][frame]:.3g}"
        )
        return (density_artist,)

    movie = animation.FuncAnimation(
        figure, update, frames=len(frame_indices), interval=1000 / fps, blit=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def save_static_outputs(
    data: Dict[str, np.ndarray],
    samples: Sequence[int],
    diagnostics: Dict[int, Dict[str, np.ndarray]],
    *,
    out_dir: Path,
    pdf_path: Path | None,
    dpi: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    def save_figure(path: Path, figure: Figure) -> None:
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        outputs.append(path)

    global_path = out_dir / "dataset_diagnostics.png"
    save_figure(global_path, make_global_figure(data, samples, diagnostics))
    for sample in samples:
        path = out_dir / f"trajectory_{sample:03d}_summary.png"
        save_figure(path, make_trajectory_figure(data, sample, diagnostics[sample]))

    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(global_path) as image:
            pdf_image = image.convert("RGB")
            pdf_image.save(
                pdf_path,
                format="PDF",
                resolution=float(dpi),
                title="Chemotactic cell-migration dataset diagnostics",
                author="visualize_chemotactic_cell_migration_data.py",
                subject="Aggregation, conservation, migration, and mesh diagnostics",
            )
            pdf_image.close()
        outputs.append(pdf_path)
    return outputs


def print_summary(
    data: Dict[str, np.ndarray],
    samples: Sequence[int],
    diagnostics: Dict[int, Dict[str, np.ndarray]],
    outputs: Sequence[Path],
) -> None:
    print("Created chemotaxis visualizations")
    print(f"  selected trajectories: {list(samples)}")
    for sample in samples:
        diag = diagnostics[sample]
        print(
            f"  trajectory {sample}: "
            f"COM shift={diag['center_of_mass_displacement'][-1]:.6g}, "
            f"<c>_n change={diag['weighted_chemo'][-1] - diag['weighted_chemo'][0]:.6g}, "
            f"peak amplification={diag['peak_density'][-1] / diag['peak_density'][0]:.6g}"
        )
    print("  outputs:")
    for path in outputs:
        print(f"    {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    data = load_archive(args.data)
    num_samples = data["rollout_states"].shape[0]
    samples = parse_trajectory_selection(args.trajectories, num_samples)
    diagnostics = {sample: trajectory_diagnostics(data, sample) for sample in samples}

    pdf_path = None if args.skip_pdf else args.pdf_path
    outputs = save_static_outputs(
        data,
        samples,
        diagnostics,
        out_dir=args.out_dir,
        pdf_path=pdf_path,
        dpi=args.dpi,
    )
    if not args.skip_gif:
        for sample in samples:
            gif_path = args.out_dir / f"trajectory_{sample:03d}_density.gif"
            make_density_gif(
                data,
                sample,
                diagnostics[sample],
                gif_path,
                max_frames=args.gif_frames,
                fps=args.gif_fps,
            )
            outputs.append(gif_path)
    print_summary(data, samples, diagnostics, outputs)


if __name__ == "__main__":
    main()
