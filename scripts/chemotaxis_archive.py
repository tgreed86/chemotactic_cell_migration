"""Load monolithic or per-trajectory chemotaxis NPZ datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REQUIRED_ARRAYS = (
    "rollout_states",
    "chemoattractant",
    "chi",
    "dt",
    "pos",
    "cell_areas",
    "boundary_cell_mask",
    "edge_index",
    "edge_attr",
    "undirected_edge_index",
)

# These flattened training-pair arrays are derivable from rollout_states and
# are not used by train.py or its plotting consumers. Omitting them avoids a
# large, unnecessary memory cost when assembling trajectory shards.
PAIR_ARRAYS = {
    "x",
    "y",
    "u_next",
    "pair_trajectory_id",
    "pair_time_index",
    "interface_flux_rate",
}

COMPATIBLE_SCALARS = (
    "dt",
    "solver_dt",
    "substeps_per_frame",
    "requested_substeps_per_frame",
    "window",
    "nx",
    "ny",
    "lx",
    "ly",
    "mesh_mode",
    "mesh_jitter",
    "num_cells",
    "num_mesh_vertices",
    "num_undirected_edges",
    "num_boundary_faces",
    "diffusion",
    "CFL",
    "reference_CFL",
    "chi_sampling",
    "chi_min",
    "chi_max",
    "target_type",
    "model",
    "equation",
    "boundary",
    "state_representation",
    "reference_refinement_levels",
    "reference_num_cells",
    "edge_attr_columns",
)


def _scalar(array: np.ndarray, name: str) -> float:
    value = np.asarray(array)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {value.shape}.")
    return float(value.item())


def _read_archive_file(path: Path, *, omit_pair_arrays: bool) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        names = (
            [name for name in archive.files if name not in PAIR_ARRAYS]
            if omit_pair_arrays
            else archive.files
        )
        data = {name: archive[name] for name in names}
    return validate_archive(data, source=path)


def validate_archive(
    data: Dict[str, np.ndarray], *, source: Path
) -> Dict[str, np.ndarray]:
    """Validate one archive and add its normalized trajectory_chi array."""
    missing = [name for name in REQUIRED_ARRAYS if name not in data]
    if missing:
        raise KeyError(
            f"Dataset {source} is missing required arrays: " + ", ".join(missing)
        )

    states = np.asarray(data["rollout_states"])
    if states.ndim != 4 or states.shape[-1] != 1:
        raise ValueError(
            f"{source}: rollout_states must have shape [trajectory, time, node, 1]."
        )
    num_trajectories, num_frames, num_nodes, _ = states.shape
    if num_frames < 2:
        raise ValueError(f"{source}: rollout_states must contain at least two frames.")

    expected_node_shapes = {
        "chemoattractant": (num_trajectories, num_nodes, 1),
        "pos": (num_trajectories, num_nodes, 2),
        "cell_areas": (num_trajectories, num_nodes),
        "boundary_cell_mask": (num_trajectories, num_nodes),
    }
    for name, expected in expected_node_shapes.items():
        if np.asarray(data[name]).shape != expected:
            raise ValueError(
                f"{source}: {name} has shape {np.asarray(data[name]).shape}; "
                f"expected {expected}."
            )

    edge_index = np.asarray(data["edge_index"])
    edge_attr = np.asarray(data["edge_attr"])
    undirected = np.asarray(data["undirected_edge_index"])
    if edge_index.ndim != 3 or edge_index.shape[0:2] != (num_trajectories, 2):
        raise ValueError(
            f"{source}: edge_index must have shape [trajectory, 2, edge]."
        )
    if edge_attr.ndim != 3 or edge_attr.shape[:2] != (
        num_trajectories,
        edge_index.shape[2],
    ):
        raise ValueError(
            f"{source}: edge_attr must have shape [trajectory, edge, channel] "
            "and align with edge_index."
        )
    if undirected.ndim != 3 or undirected.shape[0:2] != (num_trajectories, 2):
        raise ValueError(
            f"{source}: undirected_edge_index must have shape "
            "[trajectory, 2, face]."
        )
    num_faces = undirected.shape[2]
    if edge_index.shape[2] != 2 * num_faces:
        raise ValueError(
            f"{source}: expected exactly two directed graph edges per interior face."
        )
    if not np.array_equal(edge_index[:, :, :num_faces], undirected):
        raise ValueError(
            f"{source}: the first half of edge_index must match "
            "undirected_edge_index."
        )
    if not np.array_equal(edge_index[:, :, num_faces:], undirected[:, ::-1, :]):
        raise ValueError(
            f"{source}: the second half of edge_index must contain reversed "
            "interior edges."
        )
    if np.any(np.asarray(data["cell_areas"]) <= 0.0):
        raise ValueError(f"{source}: every cell area must be positive.")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"{source}: rollout_states contains non-finite values.")
    if _scalar(data["dt"], "dt") <= 0.0:
        raise ValueError(f"{source}: dt must be positive.")

    chi = np.asarray(data["chi"], dtype=np.float64)
    if chi.ndim == 0:
        chi = np.full(num_trajectories, float(chi), dtype=np.float64)
    if chi.shape != (num_trajectories,):
        raise ValueError(
            f"{source}: chi must be scalar or shape [{num_trajectories}]; "
            f"got {chi.shape}."
        )
    if np.any(chi < 0.0) or not np.all(np.isfinite(chi)):
        raise ValueError(f"{source}: chi values must be finite and nonnegative.")
    data["trajectory_chi"] = chi
    return data


def _directory_records(path: Path) -> List[Tuple[Path, int, int, int]]:
    """Return ordered (path, local count, global id, collection count) records."""
    archive_paths = sorted(
        candidate for candidate in path.glob("*.npz") if candidate.is_file()
    )
    if not archive_paths:
        raise FileNotFoundError(f"Dataset directory contains no .npz files: {path}")

    shard_records: List[Tuple[Path, int, int, int]] = []
    plain_records: List[Tuple[Path, int, int, int]] = []
    for index, archive_path in enumerate(archive_paths):
        with np.load(archive_path, allow_pickle=False) as archive:
            local_count = (
                int(np.asarray(archive["num_samples"]).item())
                if "num_samples" in archive
                else int(np.asarray(archive["rollout_states"]).shape[0])
            )
            layout = (
                str(np.asarray(archive["generation_layout"]).item())
                if "generation_layout" in archive
                else ""
            )
            if layout:
                if layout != "one_trajectory_per_file" or local_count != 1:
                    raise ValueError(
                        f"Unsupported sharded dataset layout in {archive_path}: "
                        f"{layout!r} with {local_count} trajectories."
                    )
                if "global_trajectory_id" not in archive:
                    raise KeyError(f"Shard is missing global_trajectory_id: {archive_path}")
                global_id = int(np.asarray(archive["global_trajectory_id"]).item())
                collection_count = int(
                    np.asarray(archive["collection_num_trajectories"]).item()
                )
                shard_records.append(
                    (archive_path, local_count, global_id, collection_count)
                )
            else:
                plain_records.append(
                    (archive_path, local_count, index, len(archive_paths))
                )

    # Generated shards carry explicit collection metadata. Prefer them over
    # unrelated combined archives that happen to live in the same directory.
    records = shard_records if shard_records else plain_records
    if not shard_records and any(record[1] != 1 for record in records):
        raise ValueError(
            "A dataset directory must contain one trajectory per NPZ file. "
            "Use a combined multi-trajectory NPZ as a file input instead."
        )
    if shard_records:
        collection_counts = {record[3] for record in records}
        if len(collection_counts) != 1:
            raise ValueError("Trajectory shards disagree about collection size.")
        expected_count = collection_counts.pop()
        global_ids = [record[2] for record in records]
        if len(records) != expected_count or sorted(global_ids) != list(
            range(expected_count)
        ):
            raise ValueError(
                f"Incomplete trajectory shard collection in {path}: expected IDs "
                f"0..{expected_count - 1}, found {sorted(global_ids)}."
            )
        records.sort(key=lambda record: record[2])
    return records


def _assert_compatible(
    reference: Dict[str, np.ndarray],
    current: Dict[str, np.ndarray],
    *,
    reference_path: Path,
    current_path: Path,
    trajectory_names: Sequence[str],
) -> None:
    current_trajectory_names = {
        name
        for name, value in current.items()
        if name not in PAIR_ARRAYS
        and np.asarray(value).ndim > 0
        and np.asarray(value).shape[0] == current["rollout_states"].shape[0]
    }
    if set(trajectory_names) != current_trajectory_names:
        missing = sorted(set(trajectory_names) - current_trajectory_names)
        extra = sorted(current_trajectory_names - set(trajectory_names))
        raise ValueError(
            f"Trajectory-array schema differs between {reference_path} and "
            f"{current_path}; missing={missing}, extra={extra}."
        )
    for name in trajectory_names:
        reference_value = np.asarray(reference[name])
        current_value = np.asarray(current[name])
        if reference_value.shape[1:] != current_value.shape[1:]:
            raise ValueError(
                f"{current_path}: {name} has trailing shape "
                f"{current_value.shape[1:]}; expected {reference_value.shape[1:]}."
            )
    for name in COMPATIBLE_SCALARS:
        if (name in reference) != (name in current):
            raise ValueError(
                f"Archive metadata {name!r} is not present consistently across shards."
            )
        if name in reference and not np.array_equal(reference[name], current[name]):
            raise ValueError(
                f"Incompatible {name!r} metadata between {reference_path} and "
                f"{current_path}."
            )


def _load_archive_directory(path: Path) -> Dict[str, np.ndarray]:
    records = _directory_records(path)
    total_trajectories = sum(record[1] for record in records)
    combined: Dict[str, np.ndarray] = {}
    trajectory_storage: Dict[str, np.ndarray] = {}
    trajectory_names: List[str] = []
    reference_data: Dict[str, np.ndarray] = {}
    reference_path = records[0][0]
    offset = 0
    maximum_mass_error = 0.0
    maximum_flux_error = 0.0

    for archive_path, local_count, _, _ in records:
        data = _read_archive_file(archive_path, omit_pair_arrays=True)
        if not trajectory_names:
            trajectory_names = sorted(
                name
                for name, value in data.items()
                if name not in PAIR_ARRAYS
                and np.asarray(value).ndim > 0
                and np.asarray(value).shape[0] == local_count
            )
            reference_data = data
            combined = {
                name: value for name, value in data.items() if name not in trajectory_names
            }
            for name in trajectory_names:
                value = np.asarray(data[name])
                trajectory_storage[name] = np.empty(
                    (total_trajectories,) + value.shape[1:], dtype=value.dtype
                )
        else:
            _assert_compatible(
                reference_data,
                data,
                reference_path=reference_path,
                current_path=archive_path,
                trajectory_names=trajectory_names,
            )
        stop = offset + local_count
        for name in trajectory_names:
            trajectory_storage[name][offset:stop] = data[name]
        offset = stop
        if "max_abs_mass_error" in data:
            maximum_mass_error = max(
                maximum_mass_error, float(np.asarray(data["max_abs_mass_error"]).item())
            )
        if "max_abs_interface_flux_balance_error" in data:
            maximum_flux_error = max(
                maximum_flux_error,
                float(
                    np.asarray(data["max_abs_interface_flux_balance_error"]).item()
                ),
            )

    combined.update(trajectory_storage)
    combined["num_samples"] = np.array(total_trajectories, dtype=np.int64)
    combined["num_pairs"] = np.array(
        total_trajectories * (combined["rollout_states"].shape[1] - 1),
        dtype=np.int64,
    )
    combined["global_trajectory_id"] = np.array(
        [record[2] for record in records], dtype=np.int64
    )
    combined["trajectory_source_file"] = np.array(
        [str(record[0]) for record in records]
    )
    if "collection_num_unique_meshes" in combined:
        combined["num_unique_meshes"] = np.asarray(
            combined["collection_num_unique_meshes"]
        )
    if "collection_num_unique_chemoattractant_landscapes" in combined:
        combined["num_unique_chemoattractant_landscapes"] = np.asarray(
            combined["collection_num_unique_chemoattractant_landscapes"]
        )
    chi = np.asarray(combined["trajectory_chi"], dtype=np.float64)
    combined["chi_shared_across_trajectories"] = np.array(
        np.all(chi == chi[0]), dtype=np.bool_
    )
    combined["num_unique_chi_values"] = np.array(
        np.unique(chi).size, dtype=np.int64
    )
    combined["chi_sample_min"] = np.array(np.min(chi), dtype=np.float64)
    combined["chi_sample_max"] = np.array(np.max(chi), dtype=np.float64)
    combined["cell_area_min"] = np.array(
        np.min(combined["cell_areas"]), dtype=np.float64
    )
    combined["cell_area_mean"] = np.array(
        np.mean(combined["cell_areas"], dtype=np.float64), dtype=np.float64
    )
    combined["cell_area_max"] = np.array(
        np.max(combined["cell_areas"]), dtype=np.float64
    )
    combined["density_min_observed"] = np.array(
        np.min(combined["rollout_states"]), dtype=np.float64
    )
    combined["density_max_observed"] = np.array(
        np.max(combined["rollout_states"]), dtype=np.float64
    )
    combined["max_abs_mass_error"] = np.array(
        maximum_mass_error, dtype=np.float64
    )
    combined["max_abs_interface_flux_balance_error"] = np.array(
        maximum_flux_error, dtype=np.float64
    )
    return validate_archive(combined, source=path)


def load_archive(path: Path) -> Dict[str, np.ndarray]:
    """Load one dataset archive or a compatible directory of trajectory shards."""
    path = Path(path)
    if path.is_file():
        return _read_archive_file(path, omit_pair_arrays=False)
    if path.is_dir():
        return _load_archive_directory(path)
    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def split_trajectories(
    num_trajectories: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Create deterministic, mutually exclusive trajectory splits."""
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions <= 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            "train_fraction, validation_fraction, and test_fraction must be "
            "positive and sum to one."
        )
    if num_trajectories < 3:
        raise ValueError("At least three trajectories are required for three splits.")

    order = np.random.default_rng(seed).permutation(num_trajectories)
    train_count = max(1, int(np.floor(train_fraction * num_trajectories)))
    validation_count = max(
        1, int(np.floor(validation_fraction * num_trajectories))
    )
    if train_count + validation_count >= num_trajectories:
        validation_count = 1
        train_count = num_trajectories - 2
    test_count = num_trajectories - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("The requested fractions produced an empty split.")
    return {
        "train": np.sort(order[:train_count]),
        "validation": np.sort(order[train_count : train_count + validation_count]),
        "test": np.sort(order[train_count + validation_count :]),
    }
