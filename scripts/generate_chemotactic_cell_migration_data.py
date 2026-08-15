#!/usr/bin/env python3
"""Generate conservative chemotactic cell-migration data on 2D meshes.

The prescribed-chemoattractant drift-diffusion model is

    partial_t n + div(J) = 0,
    J = chi * n * grad(c) - D * grad(n),

where ``n`` is cell density, ``c`` is a fixed chemoattractant field for each
trajectory, ``chi`` is chemotactic sensitivity, and ``D`` is diffusivity.
The square domain has no-flux boundaries, so the finite-volume total

    sum_i n_i * cell_area_i

is conserved up to floating-point roundoff.

Graph nodes are triangular finite-volume cells. By default, every trajectory
has its own independently perturbed mesh; ``--shared-mesh`` instead reuses one
mesh for all trajectories. Either way, mesh geometry remains fixed throughout
each trajectory. Graph edges connect triangles that share a face. The default
mesh is a Delaunay triangulation of a perturbed point lattice;
``--mesh-mode jittered_triangles`` provides a NumPy-only alternative with
randomized diagonals. Fluxes are evaluated once per shared face and applied
with equal and opposite signs to the two neighboring cells.

Reference trajectories are solved on uniformly refined versions of the saved
coarse meshes. The reference method uses oriented face normals, limited linear
reconstruction, a non-orthogonal diffusion correction, and SSP-RK2 time
integration. Fine-cell masses are summed into their parent coarse triangles,
so every stored state is a conservative coarse-cell average rather than a
point sample.

Example:

    python scripts/generate_chemotactic_cell_migration_data.py \
        --nx 17 --ny 17 \
        --num-samples 100 \
        --window 128 \
        --substeps-per-frame 4 \
        --out data/rollout/chemotaxis_train.npz

To make trajectories differ only through their initial cell densities, add
``--shared-mesh --shared-chemoattractant`` and use ``--chi VALUE`` (rather
than ``--chi-range``).

Important output arrays:

    x: [num_samples * window, num_cells, 4]
        Channels are [cell_density, chemoattractant,
        drift_velocity_x, drift_velocity_y]. Models without physics input can
        use x[..., :1].
    y, u_next: [num_samples * window, num_cells, 1]
    rollout_states: [num_samples, window + 1, num_cells, 1]
    chemoattractant: [num_samples, num_cells, 1]
    chemo_gradient, drift_velocity: [num_samples, num_cells, 2]
    chi: [num_samples]
        Chemotactic sensitivity sampled once per trajectory and held constant
        across space and time. With ``--chi``, every entry is identical.
    pos: [num_samples, num_cells, 2]
    edge_index: [num_samples, 2, num_directed_edges]
    edge_attr: [num_samples, num_directed_edges, 13]
    interface_flux_transport: [num_samples, window, num_undirected_edges, 1]
        Signed mass transported during each stored step, oriented from the
        first to the second cell in ``undirected_edge_index``. These labels are
        integrated from the refined SSP-RK2 reference fluxes.
    interface_flux_rate: interface_flux_transport divided by stored-frame dt
    mesh_vertices, triangles, cell_areas, shared_face_lengths
        All mesh arrays have a leading trajectory dimension.
    total_mass: [num_samples, window + 1]

For a flattened pair ``p``, select its static mesh with
``pair_trajectory_id[p]``; the same mesh is used at every time in that
trajectory.

The saved ``dt`` is the time between stored frames. ``solver_dt`` is the
refined-reference SSP-RK2 substep, and ``substeps_per_frame`` records the
actual number of reference substeps making one supervised transition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


MESH_MODES = ("delaunay", "jittered_triangles")
TARGET_TYPES = ("increment", "rate", "next")


def make_perturbed_vertices(
    *, nx: int, ny: int, lx: float, ly: float, jitter: float, rng: np.random.Generator,
) -> np.ndarray:
    """Create a quasi-uniform point set with a fixed rectangular boundary."""
    if nx < 3 or ny < 3:
        raise ValueError(f"nx and ny must both be >= 3, got {nx}, {ny}.")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError(f"lx and ly must be positive, got {lx}, {ly}.")
    if not (0.0 <= jitter < 0.45):
        raise ValueError("mesh_jitter must be in [0, 0.45).")

    x = np.linspace(0.0, lx, nx, dtype=np.float64)
    y = np.linspace(0.0, ly, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.stack([xx.ravel(), yy.ravel()], axis=1)

    hx = lx / (nx - 1)
    hy = ly / (ny - 1)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            node = j * nx + i
            vertices[node, 0] += rng.uniform(-jitter, jitter) * hx
            vertices[node, 1] += rng.uniform(-jitter, jitter) * hy
    return vertices


def _jittered_grid_triangles(
    *, nx: int, ny: int, rng: np.random.Generator
) -> np.ndarray:
    """Triangulate each perturbed grid cell using a random diagonal."""
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            lower_left = j * nx + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx
            upper_right = upper_left + 1
            if rng.random() < 0.5:
                triangles.append((lower_left, lower_right, upper_right))
                triangles.append((lower_left, upper_right, upper_left))
            else:
                triangles.append((lower_left, lower_right, upper_left))
                triangles.append((lower_right, upper_right, upper_left))
    return np.asarray(triangles, dtype=np.int64)


def triangulate_vertices(
    *, vertices: np.ndarray, nx: int, ny: int, mesh_mode: str, rng: np.random.Generator,
) -> np.ndarray:
    """Construct an irregular triangular mesh and orient cells CCW."""
    if mesh_mode == "delaunay":
        try:
            from scipy.spatial import Delaunay
        except ImportError as exc:
            raise ImportError(
                "--mesh-mode delaunay requires SciPy. Install scipy or use "
                "--mesh-mode jittered_triangles."
            ) from exc
        triangles = np.asarray(Delaunay(vertices).simplices, dtype=np.int64)
    elif mesh_mode == "jittered_triangles":
        triangles = _jittered_grid_triangles(nx=nx, ny=ny, rng=rng)
    else:
        raise ValueError(f"mesh_mode must be one of {MESH_MODES}, got {mesh_mode}.")

    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    twice_signed_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    if np.any(np.abs(twice_signed_area) <= 1.0e-14):
        raise ValueError("Mesh contains a degenerate triangle.")
    clockwise = twice_signed_area < 0.0
    if np.any(clockwise):
        triangles[clockwise, 1], triangles[clockwise, 2] = (
            triangles[clockwise, 2].copy(),
            triangles[clockwise, 1].copy(),
        )
    return triangles


def triangle_geometry(
    *, vertices: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return triangle centroids, areas, and three-point quadrature points."""
    tri_vertices = vertices[triangles]
    p0 = tri_vertices[:, 0]
    p1 = tri_vertices[:, 1]
    p2 = tri_vertices[:, 2]
    twice_area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    areas = 0.5 * twice_area
    if np.any(areas <= 0.0):
        raise ValueError("All triangles must be counterclockwise with positive area.")
    centers = np.mean(tri_vertices, axis=1)

    barycentric = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    quadrature_points = np.einsum("qv,cvd->cqd", barycentric, tri_vertices)
    return centers, areas, quadrature_points


def build_cell_graph(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build finite-volume adjacency from shared triangle faces."""
    num_cells = triangles.shape[0]
    local_faces = np.stack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],], axis=1,
    ).reshape(-1, 2)
    sorted_faces = np.sort(local_faces, axis=1)
    owners = np.repeat(np.arange(num_cells, dtype=np.int64), 3)

    order = np.lexsort((sorted_faces[:, 1], sorted_faces[:, 0]))
    sorted_faces = sorted_faces[order]
    owners = owners[order]
    unique_faces, first, counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("Mesh is non-manifold: a face has more than two owners.")

    interior_mask = counts == 2
    boundary_mask = counts == 1
    interior_faces = unique_faces[interior_mask]
    interior_first = first[interior_mask]
    cell_i = owners[interior_first]
    cell_j = owners[interior_first + 1]
    undirected_edges = np.stack([cell_i, cell_j], axis=0)

    face_vector = vertices[interior_faces[:, 1]] - vertices[interior_faces[:, 0]]
    face_lengths = np.linalg.norm(face_vector, axis=1)
    face_midpoints = 0.5 * (
        vertices[interior_faces[:, 0]] + vertices[interior_faces[:, 1]]
    )
    displacement = centers[cell_j] - centers[cell_i]
    center_distances = np.linalg.norm(displacement, axis=1)
    if np.any(face_lengths <= 0.0) or np.any(center_distances <= 0.0):
        raise ValueError("Mesh graph contains a zero-length edge.")
    face_normals = np.column_stack([face_vector[:, 1], -face_vector[:, 0]])
    face_normals /= face_lengths[:, None]
    normal_orientation = np.sum(face_normals * displacement, axis=1)
    face_normals[normal_orientation < 0.0] *= -1.0
    center_normal_distances = np.sum(face_normals * displacement, axis=1)
    if np.any(center_normal_distances <= 1.0e-14):
        raise ValueError(
            "A neighboring-cell displacement has nonpositive face-normal distance."
        )
    transmissibility = face_lengths / center_normal_distances

    boundary_faces = unique_faces[boundary_mask]
    boundary_owner = owners[first[boundary_mask]]
    boundary_vector = vertices[boundary_faces[:, 1]] - vertices[boundary_faces[:, 0]]
    boundary_lengths = np.linalg.norm(boundary_vector, axis=1)
    boundary_midpoints = 0.5 * (
        vertices[boundary_faces[:, 0]] + vertices[boundary_faces[:, 1]]
    )
    boundary_normals = np.column_stack([boundary_vector[:, 1], -boundary_vector[:, 0]])
    boundary_normals /= boundary_lengths[:, None]
    boundary_outward = boundary_midpoints - centers[boundary_owner]
    boundary_normals[np.sum(boundary_normals * boundary_outward, axis=1) < 0.0] *= -1.0
    boundary_cell_mask = np.zeros(num_cells, dtype=np.bool_)
    boundary_cell_mask[boundary_owner] = True

    sources = np.concatenate([cell_i, cell_j])
    targets = np.concatenate([cell_j, cell_i])
    directed_displacement = centers[targets] - centers[sources]
    directed_distance = np.linalg.norm(directed_displacement, axis=1)
    directed_unit = directed_displacement / directed_distance[:, None]
    directed_face_length = np.concatenate([face_lengths, face_lengths])
    directed_face_normals = np.concatenate([face_normals, -face_normals], axis=0)
    directed_normal_distance = np.concatenate(
        [center_normal_distances, center_normal_distances]
    )
    directed_transmissibility = np.concatenate([transmissibility, transmissibility])
    directed_boundary_source = boundary_cell_mask[sources].astype(np.float64)
    edge_attr = np.column_stack(
        [
            directed_displacement[:, 0],
            directed_displacement[:, 1],
            directed_distance,
            directed_unit[:, 0],
            directed_unit[:, 1],
            directed_face_length,
            directed_face_normals[:, 0],
            directed_face_normals[:, 1],
            directed_normal_distance,
            directed_transmissibility,
            areas[sources],
            areas[targets],
            directed_boundary_source,
        ]
    )

    return {
        "edge_index": np.stack([sources, targets], axis=0),
        "edge_attr": edge_attr,
        "undirected_edge_index": undirected_edges,
        "interior_faces": interior_faces,
        "shared_face_lengths": face_lengths,
        "shared_face_midpoints": face_midpoints,
        "shared_face_normals": face_normals,
        "center_distances": center_distances,
        "center_normal_distances": center_normal_distances,
        "transmissibility": transmissibility,
        "boundary_faces": boundary_faces,
        "boundary_face_owner": boundary_owner,
        "boundary_face_lengths": boundary_lengths,
        "boundary_face_midpoints": boundary_midpoints,
        "boundary_face_normals": boundary_normals,
        "boundary_cell_mask": boundary_cell_mask,
    }


def refine_triangular_mesh(
    *, vertices: np.ndarray, triangles: np.ndarray, levels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly red-refine a triangular mesh and track coarse parents."""
    if levels < 0:
        raise ValueError(f"reference_refinement_levels must be >= 0, got {levels}.")

    refined_vertices = np.asarray(vertices, dtype=np.float64).copy()
    refined_triangles = np.asarray(triangles, dtype=np.int64).copy()
    parent = np.arange(triangles.shape[0], dtype=np.int64)
    for _ in range(levels):
        edges = np.concatenate(
            [
                refined_triangles[:, [0, 1]],
                refined_triangles[:, [1, 2]],
                refined_triangles[:, [2, 0]],
            ],
            axis=0,
        )
        sorted_edges = np.sort(edges, axis=1)
        unique_edges, inverse = np.unique(sorted_edges, axis=0, return_inverse=True)
        midpoint_ids = np.arange(
            refined_vertices.shape[0],
            refined_vertices.shape[0] + unique_edges.shape[0],
            dtype=np.int64,
        )
        midpoints = 0.5 * (
            refined_vertices[unique_edges[:, 0]] + refined_vertices[unique_edges[:, 1]]
        )
        refined_vertices = np.concatenate([refined_vertices, midpoints], axis=0)

        num_triangles = refined_triangles.shape[0]
        midpoint_lookup = midpoint_ids[inverse].reshape(3, num_triangles).T
        m01 = midpoint_lookup[:, 0]
        m12 = midpoint_lookup[:, 1]
        m20 = midpoint_lookup[:, 2]
        v0 = refined_triangles[:, 0]
        v1 = refined_triangles[:, 1]
        v2 = refined_triangles[:, 2]
        refined_triangles = np.stack(
            [
                np.column_stack([v0, m01, m20]),
                np.column_stack([m01, v1, m12]),
                np.column_stack([m20, m12, v2]),
                np.column_stack([m01, m12, m20]),
            ],
            axis=1,
        ).reshape(-1, 3)
        parent = np.repeat(parent, 4)

    return refined_vertices, refined_triangles, parent


def _stack_mesh_graphs(
    graphs: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Stack equal-size per-trajectory graph dictionaries."""
    if not graphs:
        raise ValueError("At least one graph is required.")
    keys = graphs[0].keys()
    result: Dict[str, np.ndarray] = {}
    for key in keys:
        expected = graphs[0][key].shape
        if any(graph[key].shape != expected for graph in graphs[1:]):
            raise ValueError(
                f"Per-trajectory graph array {key!r} does not have a fixed shape."
            )
        result[key] = np.stack([graph[key] for graph in graphs], axis=0)
    return result


def _sample_centers(
    *,
    count: int,
    lx: float,
    ly: float,
    margin_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample source centers away from the domain boundary."""
    if not (0.0 <= margin_fraction < 0.5):
        raise ValueError("source_margin_fraction must be in [0, 0.5).")
    low = np.array([margin_fraction * lx, margin_fraction * ly])
    high = np.array([(1.0 - margin_fraction) * lx, (1.0 - margin_fraction) * ly])
    return rng.uniform(low=low, high=high, size=(count, 2))


def gaussian_mixture_value_and_gradient(
    *,
    points: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    sigmas: np.ndarray,
    baseline: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate an isotropic Gaussian mixture and its analytic gradient."""
    points = np.asarray(points, dtype=np.float64)
    delta = points[..., None, :] - centers
    sigma2 = np.square(sigmas)
    exponent = -0.5 * np.sum(np.square(delta), axis=-1) / sigma2
    components = amplitudes * np.exp(exponent)
    value = baseline + np.sum(components, axis=-1)
    gradient = np.sum(-components[..., None] * delta / sigma2[..., None], axis=-2)
    return value, gradient


def generate_chemoattractant_fields(
    *,
    num_samples: int,
    quadrature_points: np.ndarray,
    lx: float,
    ly: float,
    num_sources_min: int,
    num_sources_max: int,
    amplitude_min: float,
    amplitude_max: float,
    sigma_min: float,
    sigma_max: float,
    baseline: float,
    margin_fraction: float,
    shared_across_trajectories: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Generate fixed chemoattractant landscapes, optionally shared."""
    if num_sources_min < 1 or num_sources_max < num_sources_min:
        raise ValueError("Require 1 <= chemo_sources_min <= chemo_sources_max.")
    if not (0.0 < amplitude_min <= amplitude_max):
        raise ValueError("Require 0 < chemo_amplitude_min <= chemo_amplitude_max.")
    if not (0.0 < sigma_min <= sigma_max):
        raise ValueError("Require 0 < chemo_sigma_min <= chemo_sigma_max.")

    if quadrature_points.ndim == 3:
        num_cells = quadrature_points.shape[0]
    elif quadrature_points.ndim == 4:
        if quadrature_points.shape[0] != num_samples:
            raise ValueError("Batched quadrature_points must have one mesh per sample.")
        num_cells = quadrature_points.shape[1]
    else:
        raise ValueError(
            "quadrature_points must have shape [cells, q, 2] or "
            "[samples, cells, q, 2]."
        )
    chemo = np.empty((num_samples, num_cells), dtype=np.float64)
    chemo_gradient = np.empty((num_samples, num_cells, 2), dtype=np.float64)
    source_count = np.empty(num_samples, dtype=np.int64)
    source_centers = np.full(
        (num_samples, num_sources_max, 2), np.nan, dtype=np.float64
    )
    source_amplitudes = np.full(
        (num_samples, num_sources_max), np.nan, dtype=np.float64
    )
    source_sigmas = np.full((num_samples, num_sources_max), np.nan, dtype=np.float64)

    shared_landscape = None
    for sample in range(num_samples):
        sample_points = (
            quadrature_points
            if quadrature_points.ndim == 3
            else quadrature_points[sample]
        )
        flat_points = sample_points.reshape(-1, 2)
        if shared_landscape is None or not shared_across_trajectories:
            count = int(rng.integers(num_sources_min, num_sources_max + 1))
            centers = _sample_centers(
                count=count,
                lx=lx,
                ly=ly,
                margin_fraction=margin_fraction,
                rng=rng,
            )
            amplitudes = rng.uniform(amplitude_min, amplitude_max, size=count)
            sigmas = rng.uniform(sigma_min, sigma_max, size=count)
            if shared_across_trajectories:
                shared_landscape = (count, centers, amplitudes, sigmas)
        else:
            count, centers, amplitudes, sigmas = shared_landscape
        value_q, gradient_q = gaussian_mixture_value_and_gradient(
            points=flat_points,
            centers=centers,
            amplitudes=amplitudes,
            sigmas=sigmas,
            baseline=baseline,
        )
        chemo[sample] = np.mean(value_q.reshape(num_cells, 3), axis=1)
        chemo_gradient[sample] = np.mean(gradient_q.reshape(num_cells, 3, 2), axis=1)
        source_count[sample] = count
        source_centers[sample, :count] = centers
        source_amplitudes[sample, :count] = amplitudes
        source_sigmas[sample, :count] = sigmas

    metadata = {
        "chemo_source_count": source_count,
        "chemo_source_centers": source_centers,
        "chemo_source_amplitudes": source_amplitudes,
        "chemo_source_sigmas": source_sigmas,
    }
    return chemo, chemo_gradient, metadata


def evaluate_chemoattractant_fields(
    *, quadrature_points: np.ndarray, metadata: Dict[str, np.ndarray], baseline: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate saved chemoattractant mixtures on trajectory-specific meshes."""
    if quadrature_points.ndim != 4:
        raise ValueError("quadrature_points must have shape [samples, cells, q, 2].")
    num_samples, num_cells, num_quadrature, _ = quadrature_points.shape
    chemo = np.empty((num_samples, num_cells), dtype=np.float64)
    gradient = np.empty((num_samples, num_cells, 2), dtype=np.float64)
    for sample in range(num_samples):
        count = int(metadata["chemo_source_count"][sample])
        value_q, gradient_q = gaussian_mixture_value_and_gradient(
            points=quadrature_points[sample].reshape(-1, 2),
            centers=metadata["chemo_source_centers"][sample, :count],
            amplitudes=metadata["chemo_source_amplitudes"][sample, :count],
            sigmas=metadata["chemo_source_sigmas"][sample, :count],
            baseline=baseline,
        )
        chemo[sample] = np.mean(value_q.reshape(num_cells, num_quadrature), axis=1)
        gradient[sample] = np.mean(
            gradient_q.reshape(num_cells, num_quadrature, 2), axis=1
        )
    return chemo, gradient


def _sample_separated_center(
    *,
    lx: float,
    ly: float,
    margin_fraction: float,
    avoid: np.ndarray,
    minimum_distance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a point separated from ``avoid`` when the domain permits it."""
    candidate = _sample_centers(
        count=1, lx=lx, ly=ly, margin_fraction=margin_fraction, rng=rng,
    )[0]
    for _ in range(199):
        if np.linalg.norm(candidate - avoid) >= minimum_distance:
            break
        candidate = _sample_centers(
            count=1, lx=lx, ly=ly, margin_fraction=margin_fraction, rng=rng,
        )[0]
    return candidate


def generate_initial_density(
    *,
    num_samples: int,
    quadrature_points: np.ndarray,
    chemo_metadata: Dict[str, np.ndarray],
    lx: float,
    ly: float,
    num_blobs_min: int,
    num_blobs_max: int,
    amplitude_min: float,
    amplitude_max: float,
    sigma_min: float,
    sigma_max: float,
    background_min: float,
    background_max: float,
    margin_fraction: float,
    primary_separation: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate nonnegative Gaussian-mixture cell-density initial states."""
    if num_blobs_min < 1 or num_blobs_max < num_blobs_min:
        raise ValueError("Require 1 <= density_blobs_min <= density_blobs_max.")
    if not (0.0 < amplitude_min <= amplitude_max):
        raise ValueError("Require 0 < density_amplitude_min <= density_amplitude_max.")
    if not (0.0 < sigma_min <= sigma_max):
        raise ValueError("Require 0 < density_sigma_min <= density_sigma_max.")
    if not (0.0 <= background_min <= background_max):
        raise ValueError("Require 0 <= density_background_min <= maximum.")
    if primary_separation < 0.0:
        raise ValueError("primary_separation must be nonnegative.")

    if quadrature_points.ndim == 3:
        num_cells = quadrature_points.shape[0]
    elif quadrature_points.ndim == 4:
        if quadrature_points.shape[0] != num_samples:
            raise ValueError("Batched quadrature_points must have one mesh per sample.")
        num_cells = quadrature_points.shape[1]
    else:
        raise ValueError(
            "quadrature_points must have shape [cells, q, 2] or "
            "[samples, cells, q, 2]."
        )
    density = np.empty((num_samples, num_cells), dtype=np.float64)
    blob_count = np.empty(num_samples, dtype=np.int64)
    blob_centers = np.full((num_samples, num_blobs_max, 2), np.nan, dtype=np.float64)
    blob_amplitudes = np.full((num_samples, num_blobs_max), np.nan, dtype=np.float64)
    blob_sigmas = np.full((num_samples, num_blobs_max), np.nan, dtype=np.float64)
    backgrounds = np.empty(num_samples, dtype=np.float64)

    source_centers = chemo_metadata["chemo_source_centers"]
    source_amplitudes = chemo_metadata["chemo_source_amplitudes"]
    for sample in range(num_samples):
        sample_points = (
            quadrature_points
            if quadrature_points.ndim == 3
            else quadrature_points[sample]
        )
        flat_points = sample_points.reshape(-1, 2)
        count = int(rng.integers(num_blobs_min, num_blobs_max + 1))
        strongest_source = int(np.nanargmax(source_amplitudes[sample]))
        primary_source = source_centers[sample, strongest_source]
        centers = _sample_centers(
            count=count, lx=lx, ly=ly, margin_fraction=margin_fraction, rng=rng,
        )
        centers[0] = _sample_separated_center(
            lx=lx,
            ly=ly,
            margin_fraction=margin_fraction,
            avoid=primary_source,
            minimum_distance=primary_separation,
            rng=rng,
        )
        amplitudes = rng.uniform(amplitude_min, amplitude_max, size=count)
        sigmas = rng.uniform(sigma_min, sigma_max, size=count)
        background = float(rng.uniform(background_min, background_max))
        value_q, _ = gaussian_mixture_value_and_gradient(
            points=flat_points,
            centers=centers,
            amplitudes=amplitudes,
            sigmas=sigmas,
            baseline=background,
        )
        density[sample] = np.mean(value_q.reshape(num_cells, 3), axis=1)
        blob_count[sample] = count
        blob_centers[sample, :count] = centers
        blob_amplitudes[sample, :count] = amplitudes
        blob_sigmas[sample, :count] = sigmas
        backgrounds[sample] = background

    metadata = {
        "density_blob_count": blob_count,
        "density_blob_centers": blob_centers,
        "density_blob_amplitudes": blob_amplitudes,
        "density_blob_sigmas": blob_sigmas,
        "density_background": backgrounds,
    }
    return density, metadata


def face_drift_velocity(
    *,
    face_midpoints: np.ndarray,
    face_normals: np.ndarray,
    chemo_metadata: Dict[str, np.ndarray],
    chi: np.ndarray,
) -> np.ndarray:
    """Evaluate chi*grad(c).normal at every oriented interior face."""
    if face_midpoints.ndim != 3 or face_normals.shape != face_midpoints.shape:
        raise ValueError(
            "face_midpoints and face_normals must have shape [samples, faces, 2]."
        )
    num_samples, num_faces, _ = face_midpoints.shape
    chi = np.asarray(chi, dtype=np.float64)
    if chi.shape != (num_samples,):
        raise ValueError("chi must have shape [samples].")
    speed = np.empty((num_samples, num_faces), dtype=np.float64)
    for sample in range(num_samples):
        count = int(chemo_metadata["chemo_source_count"][sample])
        _, gradient = gaussian_mixture_value_and_gradient(
            points=face_midpoints[sample],
            centers=chemo_metadata["chemo_source_centers"][sample, :count],
            amplitudes=chemo_metadata["chemo_source_amplitudes"][sample, :count],
            sigmas=chemo_metadata["chemo_source_sigmas"][sample, :count],
            baseline=0.0,
        )
        speed[sample] = chi[sample] * np.sum(
            gradient * face_normals[sample], axis=1
        )
    return speed


def _flatten_edge_indices(
    undirected_edges: np.ndarray, num_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened batched cell indices for both ends of every edge."""
    num_samples = undirected_edges.shape[0]
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * num_cells
    return (
        sample_offsets + undirected_edges[:, 0],
        sample_offsets + undirected_edges[:, 1],
    )


def stable_explicit_timestep(
    *,
    areas: np.ndarray,
    undirected_edges: np.ndarray,
    shared_face_lengths: np.ndarray,
    center_normal_distances: np.ndarray,
    drift_speed: np.ndarray,
    diffusion: float,
    cfl: float,
) -> Tuple[float, np.ndarray]:
    """Return a positivity-preserving explicit step based on outgoing rates."""
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be nonnegative, got {diffusion}.")
    if not (0.0 < cfl <= 1.0):
        raise ValueError(f"CFL must be in (0, 1], got {cfl}.")
    if areas.ndim != 2 or undirected_edges.ndim != 3:
        raise ValueError("Mesh geometry must have a leading trajectory dimension.")
    num_samples, num_cells = areas.shape
    if drift_speed.shape[0] != num_samples:
        raise ValueError("drift_speed and areas must contain the same samples.")
    diffusive_conductance = diffusion * shared_face_lengths / center_normal_distances
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)
    outward = np.zeros(num_samples * num_cells, dtype=np.float64)
    np.add.at(
        outward,
        index_i.ravel(),
        (
            diffusive_conductance + shared_face_lengths * np.maximum(drift_speed, 0.0)
        ).ravel(),
    )
    np.add.at(
        outward,
        index_j.ravel(),
        (
            diffusive_conductance + shared_face_lengths * np.maximum(-drift_speed, 0.0)
        ).ravel(),
    )
    outward = outward.reshape(num_samples, num_cells)
    max_rates = np.max(outward / areas, axis=1)
    global_rate = float(np.max(max_rates))
    if global_rate <= 0.0:
        raise ValueError(
            "At least one of diffusion or chemotactic drift must be nonzero."
        )
    return cfl / global_rate, max_rates


def least_squares_inverse_moments(
    *, centers: np.ndarray, undirected_edges: np.ndarray
) -> np.ndarray:
    """Precompute per-cell weighted least-squares gradient matrices."""
    num_samples, num_cells, _ = centers.shape
    cell_i = undirected_edges[:, 0]
    cell_j = undirected_edges[:, 1]
    sample_index = np.arange(num_samples, dtype=np.int64)[:, None]
    displacement = centers[sample_index, cell_j] - centers[sample_index, cell_i]
    distance2 = np.sum(np.square(displacement), axis=2)
    weights = 1.0 / np.maximum(distance2, 1.0e-30)
    outer = weights[..., None, None] * np.einsum(
        "...a,...b->...ab", displacement, displacement
    )
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)
    moments = np.zeros((num_samples * num_cells, 2, 2), dtype=np.float64)
    np.add.at(moments, index_i.ravel(), outer.reshape(-1, 2, 2))
    np.add.at(moments, index_j.ravel(), outer.reshape(-1, 2, 2))
    moments = moments.reshape(num_samples, num_cells, 2, 2)
    return np.linalg.pinv(moments, rcond=1.0e-12)


def reconstruct_cell_gradients(
    values: np.ndarray,
    *,
    centers: np.ndarray,
    undirected_edges: np.ndarray,
    inverse_moments: np.ndarray,
) -> np.ndarray:
    """Reconstruct cell gradients from neighboring cell-average differences."""
    num_samples, num_cells = values.shape
    cell_i = undirected_edges[:, 0]
    cell_j = undirected_edges[:, 1]
    sample_index = np.arange(num_samples, dtype=np.int64)[:, None]
    displacement = centers[sample_index, cell_j] - centers[sample_index, cell_i]
    distance2 = np.sum(np.square(displacement), axis=2)
    delta = values[sample_index, cell_j] - values[sample_index, cell_i]
    contribution = (
        delta[..., None] * displacement / np.maximum(distance2[..., None], 1.0e-30)
    )
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)
    right_hand_side = np.zeros((num_samples * num_cells, 2), dtype=np.float64)
    np.add.at(right_hand_side, index_i.ravel(), contribution.reshape(-1, 2))
    np.add.at(right_hand_side, index_j.ravel(), contribution.reshape(-1, 2))
    right_hand_side = right_hand_side.reshape(num_samples, num_cells, 2)
    return np.einsum("...ab,...b->...a", inverse_moments, right_hand_side)


def limited_face_reconstruction(
    values: np.ndarray,
    *,
    gradients: np.ndarray,
    centers: np.ndarray,
    undirected_edges: np.ndarray,
    face_midpoints: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barth-Jespersen-limit linear states on both sides of every face."""
    num_samples, num_cells = values.shape
    cell_i = undirected_edges[:, 0]
    cell_j = undirected_edges[:, 1]
    sample_index = np.arange(num_samples, dtype=np.int64)[:, None]
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)

    minimum = values.ravel().copy()
    maximum = values.ravel().copy()
    values_i = values[sample_index, cell_i]
    values_j = values[sample_index, cell_j]
    np.minimum.at(minimum, index_i.ravel(), values_j.ravel())
    np.minimum.at(minimum, index_j.ravel(), values_i.ravel())
    np.maximum.at(maximum, index_i.ravel(), values_j.ravel())
    np.maximum.at(maximum, index_j.ravel(), values_i.ravel())
    minimum = minimum.reshape(num_samples, num_cells)
    maximum = maximum.reshape(num_samples, num_cells)

    offset_i = face_midpoints - centers[sample_index, cell_i]
    offset_j = face_midpoints - centers[sample_index, cell_j]
    increment_i = np.sum(gradients[sample_index, cell_i] * offset_i, axis=2)
    increment_j = np.sum(gradients[sample_index, cell_j] * offset_j, axis=2)

    def limiter_ratio(
        cell_values: np.ndarray,
        increments: np.ndarray,
        local_minimum: np.ndarray,
        local_maximum: np.ndarray,
    ) -> np.ndarray:
        ratio = np.ones_like(increments)
        positive = increments > 1.0e-14
        negative = increments < -1.0e-14
        ratio[positive] = (
            local_maximum[positive] - cell_values[positive]
        ) / increments[positive]
        ratio[negative] = (
            local_minimum[negative] - cell_values[negative]
        ) / increments[negative]
        return np.clip(ratio, 0.0, 1.0)

    ratio_i = limiter_ratio(
        values_i,
        increment_i,
        minimum[sample_index, cell_i],
        maximum[sample_index, cell_i],
    )
    ratio_j = limiter_ratio(
        values_j,
        increment_j,
        minimum[sample_index, cell_j],
        maximum[sample_index, cell_j],
    )
    limiter = np.ones(num_samples * num_cells, dtype=np.float64)
    np.minimum.at(limiter, index_i.ravel(), ratio_i.ravel())
    np.minimum.at(limiter, index_j.ravel(), ratio_j.ravel())
    limiter = limiter.reshape(num_samples, num_cells)

    limited_gradients = gradients * limiter[..., None]
    left_state = values_i + limiter[sample_index, cell_i] * increment_i
    right_state = values_j + limiter[sample_index, cell_j] * increment_j
    return left_state, right_state, limited_gradients


def finite_volume_face_fluxes(
    density: np.ndarray,
    *,
    undirected_edges: np.ndarray,
    shared_face_lengths: np.ndarray,
    centers: np.ndarray,
    face_midpoints: np.ndarray,
    face_normals: np.ndarray,
    center_normal_distances: np.ndarray,
    drift_speed: np.ndarray,
    diffusion: float,
    inverse_moments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return high- and low-order oriented mass fluxes at shared faces."""
    num_samples, num_cells = density.shape
    cell_i = undirected_edges[:, 0]
    cell_j = undirected_edges[:, 1]
    sample_index = np.arange(num_samples, dtype=np.int64)[:, None]
    density_i = density[sample_index, cell_i]
    density_j = density[sample_index, cell_j]
    gradients = reconstruct_cell_gradients(
        density,
        centers=centers,
        undirected_edges=undirected_edges,
        inverse_moments=inverse_moments,
    )
    left_state, right_state, limited_gradients = limited_face_reconstruction(
        density,
        gradients=gradients,
        centers=centers,
        undirected_edges=undirected_edges,
        face_midpoints=face_midpoints,
    )
    upwind_density = np.where(drift_speed >= 0.0, left_state, right_state)

    displacement = centers[sample_index, cell_j] - centers[sample_index, cell_i]
    average_gradient = 0.5 * (
        limited_gradients[sample_index, cell_i]
        + limited_gradients[sample_index, cell_j]
    )
    nonorthogonal_direction = (
        face_normals - displacement / center_normal_distances[..., None]
    )
    normal_gradient = (density_j - density_i) / center_normal_distances + np.sum(
        average_gradient * nonorthogonal_direction, axis=2
    )
    high_order_flux = drift_speed * upwind_density - diffusion * normal_gradient
    low_order_upwind = np.where(drift_speed >= 0.0, density_i, density_j)
    low_order_normal_gradient = (density_j - density_i) / center_normal_distances
    low_order_flux = (
        drift_speed * low_order_upwind - diffusion * low_order_normal_gradient
    )
    return (
        shared_face_lengths * high_order_flux,
        shared_face_lengths * low_order_flux,
    )


def _mass_change_from_face_flux(
    face_mass_flux: np.ndarray, *, undirected_edges: np.ndarray, num_cells: int,
) -> np.ndarray:
    """Apply each oriented face flux with opposite signs to its two cells."""
    num_samples = face_mass_flux.shape[0]
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)
    mass_change = np.zeros(num_samples * num_cells, dtype=np.float64)
    np.add.at(mass_change, index_i.ravel(), -face_mass_flux.ravel())
    np.add.at(mass_change, index_j.ravel(), face_mass_flux.ravel())
    return mass_change.reshape(num_samples, num_cells)


def positivity_preserving_euler_step(
    density: np.ndarray,
    *,
    dt: float,
    areas: np.ndarray,
    undirected_edges: np.ndarray,
    shared_face_lengths: np.ndarray,
    centers: np.ndarray,
    face_midpoints: np.ndarray,
    face_normals: np.ndarray,
    center_normal_distances: np.ndarray,
    drift_speed: np.ndarray,
    diffusion: float,
    inverse_moments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take one Euler step and return its oriented limited face mass flux."""
    num_samples, num_cells = density.shape
    high_flux, low_flux = finite_volume_face_fluxes(
        density,
        undirected_edges=undirected_edges,
        shared_face_lengths=shared_face_lengths,
        centers=centers,
        face_midpoints=face_midpoints,
        face_normals=face_normals,
        center_normal_distances=center_normal_distances,
        drift_speed=drift_speed,
        diffusion=diffusion,
        inverse_moments=inverse_moments,
    )
    low_mass_change = _mass_change_from_face_flux(
        low_flux, undirected_edges=undirected_edges, num_cells=num_cells,
    )
    low_state = density + dt * low_mass_change / areas
    if np.min(low_state) < -1.0e-12:
        raise RuntimeError(
            "Monotone reference stage became negative; reduce --reference-CFL."
        )

    correction_mass = dt * (high_flux - low_flux)
    cell_i = undirected_edges[:, 0]
    cell_j = undirected_edges[:, 1]
    index_i, index_j = _flatten_edge_indices(undirected_edges, num_cells)
    requested_loss = np.zeros(num_samples * num_cells, dtype=np.float64)
    np.add.at(
        requested_loss, index_i.ravel(), np.maximum(correction_mass, 0.0).ravel(),
    )
    np.add.at(
        requested_loss, index_j.ravel(), np.maximum(-correction_mass, 0.0).ravel(),
    )
    requested_loss = requested_loss.reshape(num_samples, num_cells)
    available_mass = np.maximum(low_state, 0.0) * areas
    limiter = np.ones_like(available_mass)
    active = requested_loss > 0.0
    limiter[active] = np.minimum(1.0, available_mass[active] / requested_loss[active])
    sample_index = np.arange(num_samples, dtype=np.int64)[:, None]
    edge_limiter = np.where(
        correction_mass >= 0.0,
        limiter[sample_index, cell_i],
        limiter[sample_index, cell_j],
    )
    limited_flux = low_flux + edge_limiter * (high_flux - low_flux)
    mass_change = _mass_change_from_face_flux(
        limited_flux, undirected_edges=undirected_edges, num_cells=num_cells,
    )
    return density + dt * mass_change / areas, limited_flux


def advance_drift_diffusion_ssprk2(
    density: np.ndarray,
    *,
    dt: float,
    areas: np.ndarray,
    undirected_edges: np.ndarray,
    shared_face_lengths: np.ndarray,
    centers: np.ndarray,
    face_midpoints: np.ndarray,
    face_normals: np.ndarray,
    center_normal_distances: np.ndarray,
    drift_speed: np.ndarray,
    diffusion: float,
    inverse_moments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance one SSP-RK2 substep and return integrated face transport."""
    kwargs = {
        "areas": areas,
        "undirected_edges": undirected_edges,
        "shared_face_lengths": shared_face_lengths,
        "centers": centers,
        "face_midpoints": face_midpoints,
        "face_normals": face_normals,
        "center_normal_distances": center_normal_distances,
        "drift_speed": drift_speed,
        "diffusion": diffusion,
        "inverse_moments": inverse_moments,
    }
    stage_one, stage_one_flux = positivity_preserving_euler_step(
        density, dt=dt, **kwargs
    )
    if np.min(stage_one) < -1.0e-10:
        raise RuntimeError(
            "Reference SSP-RK2 stage became negative; reduce --reference-CFL."
        )
    stage_two_euler, stage_two_flux = positivity_preserving_euler_step(
        stage_one, dt=dt, **kwargs
    )
    result = 0.5 * density + 0.5 * stage_two_euler
    if np.min(result) < -1.0e-10:
        raise RuntimeError(
            "Reference SSP-RK2 state became negative; reduce --reference-CFL."
        )
    integrated_face_transport = 0.5 * dt * (stage_one_flux + stage_two_flux)
    return result, integrated_face_transport


def project_refined_to_coarse(
    refined_values: np.ndarray,
    *,
    refined_areas: np.ndarray,
    coarse_parent: np.ndarray,
    coarse_areas: np.ndarray,
) -> np.ndarray:
    """Conservatively project refined cell averages to coarse parents."""
    num_samples, num_refined_cells = refined_values.shape
    num_coarse_cells = coarse_areas.shape[1]
    sample_offsets = np.arange(num_samples, dtype=np.int64)[:, None] * num_coarse_cells
    parent_index = sample_offsets + coarse_parent
    coarse_mass = np.zeros(num_samples * num_coarse_cells, dtype=np.float64)
    np.add.at(
        coarse_mass,
        parent_index.ravel(),
        (refined_values * refined_areas).reshape(num_samples * num_refined_cells),
    )
    return coarse_mass.reshape(num_samples, num_coarse_cells) / coarse_areas


def refined_to_coarse_face_map(
    *,
    refined_edges: np.ndarray,
    refined_parent: np.ndarray,
    coarse_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map refined interface orientations onto their parent coarse faces."""
    num_samples, _, num_refined_faces = refined_edges.shape
    if refined_parent.shape[0] != num_samples or coarse_edges.shape[0] != num_samples:
        raise ValueError("Refined/coarse face-map arrays must share a sample axis.")
    coarse_face_index = np.full(
        (num_samples, num_refined_faces), -1, dtype=np.int64
    )
    orientation = np.zeros((num_samples, num_refined_faces), dtype=np.float64)
    for sample in range(num_samples):
        lookup = {}
        for face, (cell_i, cell_j) in enumerate(coarse_edges[sample].T):
            lookup[(int(cell_i), int(cell_j))] = (face, 1.0)
            lookup[(int(cell_j), int(cell_i))] = (face, -1.0)
        fine_i, fine_j = refined_edges[sample]
        parent_i = refined_parent[sample, fine_i]
        parent_j = refined_parent[sample, fine_j]
        for fine_face, (cell_i, cell_j) in enumerate(zip(parent_i, parent_j)):
            if cell_i == cell_j:
                continue
            mapping = lookup.get((int(cell_i), int(cell_j)))
            if mapping is None:
                raise RuntimeError(
                    "A refined cross-parent face has no corresponding coarse face."
                )
            coarse_face_index[sample, fine_face], orientation[sample, fine_face] = mapping
    return coarse_face_index, orientation


def aggregate_refined_face_transport(
    refined_transport: np.ndarray,
    *,
    coarse_face_index: np.ndarray,
    orientation: np.ndarray,
    num_coarse_faces: int,
) -> np.ndarray:
    """Sum oriented refined transports across each parent coarse face."""
    if refined_transport.shape != coarse_face_index.shape:
        raise ValueError("Refined transport and face-map shapes must match.")
    coarse_transport = np.zeros(
        (refined_transport.shape[0], num_coarse_faces), dtype=np.float64
    )
    for sample in range(refined_transport.shape[0]):
        active = coarse_face_index[sample] >= 0
        np.add.at(
            coarse_transport[sample],
            coarse_face_index[sample, active],
            orientation[sample, active] * refined_transport[sample, active],
        )
    return coarse_transport


def generate_rollouts(
    *,
    refined_density0: np.ndarray,
    refined_areas: np.ndarray,
    coarse_parent: np.ndarray,
    coarse_areas: np.ndarray,
    coarse_undirected_edges: np.ndarray,
    undirected_edges: np.ndarray,
    shared_face_lengths: np.ndarray,
    centers: np.ndarray,
    face_midpoints: np.ndarray,
    face_normals: np.ndarray,
    center_normal_distances: np.ndarray,
    drift_speed: np.ndarray,
    diffusion: float,
    solver_dt: float,
    window: int,
    substeps_per_frame: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate density trajectories and exact coarse-face transport labels."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")
    if substeps_per_frame <= 0:
        raise ValueError(
            f"substeps_per_frame must be positive, got {substeps_per_frame}."
        )
    num_samples = refined_density0.shape[0]
    num_cells = coarse_areas.shape[1]
    rollout = np.empty((num_samples, window + 1, num_cells), dtype=np.float64)
    rollout[:, 0] = project_refined_to_coarse(
        refined_density0,
        refined_areas=refined_areas,
        coarse_parent=coarse_parent,
        coarse_areas=coarse_areas,
    )
    state = refined_density0.copy()
    inverse_moments = least_squares_inverse_moments(
        centers=centers, undirected_edges=undirected_edges
    )
    coarse_face_index, orientation = refined_to_coarse_face_map(
        refined_edges=undirected_edges,
        refined_parent=coarse_parent,
        coarse_edges=coarse_undirected_edges,
    )
    num_coarse_faces = coarse_undirected_edges.shape[2]
    interface_flux_transport = np.zeros(
        (num_samples, window, num_coarse_faces), dtype=np.float64
    )
    for frame in range(window):
        frame_transport = np.zeros(
            (num_samples, undirected_edges.shape[2]), dtype=np.float64
        )
        for _ in range(substeps_per_frame):
            state, substep_transport = advance_drift_diffusion_ssprk2(
                state,
                areas=refined_areas,
                undirected_edges=undirected_edges,
                shared_face_lengths=shared_face_lengths,
                centers=centers,
                face_midpoints=face_midpoints,
                face_normals=face_normals,
                center_normal_distances=center_normal_distances,
                drift_speed=drift_speed,
                diffusion=diffusion,
                dt=solver_dt,
                inverse_moments=inverse_moments,
            )
            frame_transport += substep_transport
        interface_flux_transport[:, frame] = aggregate_refined_face_transport(
            frame_transport,
            coarse_face_index=coarse_face_index,
            orientation=orientation,
            num_coarse_faces=num_coarse_faces,
        )
        rollout[:, frame + 1] = project_refined_to_coarse(
            state,
            refined_areas=refined_areas,
            coarse_parent=coarse_parent,
            coarse_areas=coarse_areas,
        )
    return rollout, interface_flux_transport


def _metadata_to_npz(metadata: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Convert Python metadata values into arrays accepted by np.savez."""
    result: Dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            result[key] = np.array(value)
        elif isinstance(value, bool):
            result[key] = np.array(value, dtype=np.bool_)
        elif isinstance(value, int):
            result[key] = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            result[key] = np.array(value, dtype=np.float64)
        else:
            result[key] = np.asarray(value)
    return result


def save_dataset(
    *,
    out_path: Path,
    rollout: np.ndarray,
    interface_flux_transport: np.ndarray,
    chemo: np.ndarray,
    chemo_gradient: np.ndarray,
    drift_velocity: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    graph: Dict[str, np.ndarray],
    frame_dt: float,
    target_type: str,
    metadata: Dict[str, object],
) -> None:
    """Flatten rollout pairs and write a compressed NumPy dataset."""
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {TARGET_TYPES}.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples, num_times, num_cells = rollout.shape
    window = num_times - 1
    num_pairs = num_samples * window
    density_now = rollout[:, :-1].reshape(num_pairs, num_cells)
    density_next = rollout[:, 1:].reshape(num_pairs, num_cells)
    pair_trajectory_id = np.repeat(np.arange(num_samples, dtype=np.int64), window)
    pair_time_index = np.tile(np.arange(window, dtype=np.int64), num_samples)

    chemo_pair = chemo[pair_trajectory_id]
    velocity_pair = drift_velocity[pair_trajectory_id]
    x = np.concatenate(
        [density_now[..., None], chemo_pair[..., None], velocity_pair], axis=2
    )
    if target_type == "increment":
        target = density_next - density_now
    elif target_type == "rate":
        target = (density_next - density_now) / frame_dt
    else:
        target = density_next
    total_mass = np.sum(rollout * areas[:, None, :], axis=2)

    np.savez_compressed(
        out_path,
        x=x.astype(np.float32),
        y=target[..., None].astype(np.float32),
        u_next=density_next[..., None].astype(np.float32),
        pair_trajectory_id=pair_trajectory_id,
        pair_time_index=pair_time_index,
        rollout_states=rollout[..., None].astype(np.float32),
        interface_flux_transport=interface_flux_transport[..., None].astype(
            np.float32
        ),
        interface_flux_rate=(interface_flux_transport / frame_dt)[..., None].astype(
            np.float32
        ),
        chemoattractant=chemo[..., None].astype(np.float32),
        chemo_gradient=chemo_gradient.astype(np.float32),
        drift_velocity=drift_velocity.astype(np.float32),
        pos=centers.astype(np.float32),
        edge_index=graph["edge_index"].astype(np.int64),
        edge_attr=graph["edge_attr"].astype(np.float32),
        mesh_vertices=vertices.astype(np.float64),
        triangles=triangles.astype(np.int64),
        cell_centers=centers.astype(np.float64),
        cell_areas=areas.astype(np.float64),
        undirected_edge_index=graph["undirected_edge_index"].astype(np.int64),
        interior_faces=graph["interior_faces"].astype(np.int64),
        shared_face_lengths=graph["shared_face_lengths"].astype(np.float64),
        shared_face_midpoints=graph["shared_face_midpoints"].astype(np.float64),
        shared_face_normals=graph["shared_face_normals"].astype(np.float64),
        center_distances=graph["center_distances"].astype(np.float64),
        center_normal_distances=graph["center_normal_distances"].astype(np.float64),
        transmissibility=graph["transmissibility"].astype(np.float64),
        boundary_faces=graph["boundary_faces"].astype(np.int64),
        boundary_face_owner=graph["boundary_face_owner"].astype(np.int64),
        boundary_face_lengths=graph["boundary_face_lengths"].astype(np.float64),
        boundary_face_midpoints=graph["boundary_face_midpoints"].astype(np.float64),
        boundary_face_normals=graph["boundary_face_normals"].astype(np.float64),
        boundary_cell_mask=graph["boundary_cell_mask"],
        total_mass=total_mass.astype(np.float64),
        **_metadata_to_npz(metadata),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conservative prescribed-chemoattractant drift-diffusion "
            "data on trajectory-specific irregular 2D triangular meshes."
        )
    )
    parser.add_argument("--nx", type=int, default=17, help="Mesh points in x.")
    parser.add_argument("--ny", type=int, default=17, help="Mesh points in y.")
    parser.add_argument("--lx", type=float, default=1.0, help="Domain x length.")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain y length.")
    parser.add_argument(
        "--mesh-mode",
        choices=MESH_MODES,
        default="delaunay",
        help="Triangulation method.",
    )
    parser.add_argument(
        "--mesh-jitter",
        type=float,
        default=0.30,
        help="Interior point displacement as a fraction of nominal spacing.",
    )
    parser.add_argument(
        "--shared-mesh",
        action="store_true",
        help=(
            "Generate one mesh and reuse it for every trajectory. By default, "
            "each trajectory receives a unique mesh."
        ),
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of trajectories."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=128,
        help="Number of stored transitions per trajectory.",
    )
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=4,
        help=(
            "Minimum reference substeps per stored transition; automatically "
            "increased when refined-mesh stability requires it."
        ),
    )
    parser.add_argument(
        "--reference-refinement-levels",
        type=int,
        default=1,
        help=(
            "Uniform red-refinement levels for the reference solve; each "
            "level creates four child triangles per cell."
        ),
    )
    parser.add_argument(
        "--reference-cfl",
        type=float,
        default=0.80,
        help="Stability factor for the refined MUSCL/SSP-RK2 reference solve.",
    )
    parser.add_argument(
        "--diffusion", type=float, default=1.0e-2, help="Cell diffusivity D."
    )
    chi_group = parser.add_mutually_exclusive_group()
    chi_group.add_argument(
        "--chi",
        type=float,
        default=0.1,
        help="Fixed chemotactic sensitivity chi for every trajectory (default: 0.1).",
    )
    chi_group.add_argument(
        "--chi-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help=(
            "Sample one constant chi per trajectory uniformly from [MIN, MAX]. "
            "The sampled value is fixed across space and time within that trajectory."
        ),
    )
    parser.add_argument(
        "--CFL", type=float, default=0.45, help="Explicit stability factor."
    )
    parser.add_argument(
        "--solver-dt",
        type=float,
        default=None,
        help=(
            "Optional nominal coarse-grid substep used to set stored-frame "
            "dt. The refined reference is automatically subcycled."
        ),
    )
    parser.add_argument(
        "--chemo-sources-min", type=int, default=1, help="Minimum attractors."
    )
    parser.add_argument(
        "--chemo-sources-max", type=int, default=3, help="Maximum attractors."
    )
    parser.add_argument("--chemo-amplitude-min", type=float, default=0.5)
    parser.add_argument("--chemo-amplitude-max", type=float, default=1.5)
    parser.add_argument("--chemo-sigma-min", type=float, default=0.10)
    parser.add_argument("--chemo-sigma-max", type=float, default=0.22)
    parser.add_argument("--chemo-baseline", type=float, default=0.0)
    parser.add_argument("--source-margin-fraction", type=float, default=0.08)
    parser.add_argument(
        "--shared-chemoattractant",
        action="store_true",
        help=(
            "Use the same Gaussian source count, locations, amplitudes, and "
            "widths for every trajectory. By default, each trajectory samples "
            "a separate chemoattractant landscape."
        ),
    )
    parser.add_argument("--density-blobs-min", type=int, default=1)
    parser.add_argument("--density-blobs-max", type=int, default=3)
    parser.add_argument("--density-amplitude-min", type=float, default=0.5)
    parser.add_argument("--density-amplitude-max", type=float, default=1.5)
    parser.add_argument("--density-sigma-min", type=float, default=0.06)
    parser.add_argument("--density-sigma-max", type=float, default=0.14)
    parser.add_argument("--density-background-min", type=float, default=0.01)
    parser.add_argument("--density-background-max", type=float, default=0.05)
    parser.add_argument("--density-margin-fraction", type=float, default=0.08)
    parser.add_argument(
        "--primary-separation",
        type=float,
        default=0.25,
        help="Minimum initial separation from the strongest attractor when possible.",
    )
    parser.add_argument(
        "--target-type",
        choices=TARGET_TYPES,
        default="increment",
        help="Contents of y.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if args.substeps_per_frame <= 0:
        raise ValueError("substeps_per_frame must be positive.")
    if args.reference_refinement_levels < 0:
        raise ValueError("reference_refinement_levels must be nonnegative.")
    if not (0.0 < args.reference_cfl <= 1.0):
        raise ValueError("reference_cfl must be in (0, 1].")
    if args.chi_range is None:
        if args.chi < 0.0:
            raise ValueError("chi must be nonnegative.")
    else:
        chi_min, chi_max = args.chi_range
        if chi_min < 0.0 or chi_max < 0.0:
            raise ValueError("chi-range endpoints must be nonnegative.")
        if chi_min > chi_max:
            raise ValueError("chi-range MIN must not exceed MAX.")
    if args.chemo_baseline < 0.0:
        raise ValueError("chemo_baseline must be nonnegative.")
    if args.num_samples > 1 and not args.shared_mesh and args.mesh_jitter <= 0.0:
        raise ValueError(
            "mesh_jitter must be positive when num_samples > 1 so every "
            "trajectory can have unique geometry."
        )

    rng = np.random.default_rng(args.seed)
    if args.chi_range is None:
        trajectory_chi = np.full(args.num_samples, args.chi, dtype=np.float64)
        chi_sampling = "fixed"
        chi_distribution_min = float(args.chi)
        chi_distribution_max = float(args.chi)
    else:
        chi_min, chi_max = args.chi_range
        trajectory_chi = rng.uniform(chi_min, chi_max, size=args.num_samples)
        chi_sampling = "uniform_per_trajectory"
        chi_distribution_min = float(chi_min)
        chi_distribution_max = float(chi_max)
    domain_area = float(args.lx * args.ly)
    vertices_list = []
    triangles_list = []
    centers_list = []
    areas_list = []
    quadrature_list = []
    graphs = []
    reference_vertices_list = []
    reference_triangles_list = []
    reference_centers_list = []
    reference_areas_list = []
    reference_quadrature_list = []
    reference_parents_list = []
    reference_graphs = []
    mesh_signatures = set()

    for sample in range(args.num_samples):
        if sample > 0 and args.shared_mesh:
            vertices_sample = vertices_list[0]
            triangles_sample = triangles_list[0]
            centers_sample = centers_list[0]
            areas_sample = areas_list[0]
            quadrature_sample = quadrature_list[0]
            graph_sample = graphs[0]
            reference_vertices_sample = reference_vertices_list[0]
            reference_triangles_sample = reference_triangles_list[0]
            reference_centers_sample = reference_centers_list[0]
            reference_areas_sample = reference_areas_list[0]
            reference_quadrature_sample = reference_quadrature_list[0]
            reference_parent_sample = reference_parents_list[0]
            reference_graph_sample = reference_graphs[0]
        else:
            vertices_sample = make_perturbed_vertices(
                nx=args.nx,
                ny=args.ny,
                lx=args.lx,
                ly=args.ly,
                jitter=args.mesh_jitter,
                rng=rng,
            )
            signature = vertices_sample.tobytes()
            if signature in mesh_signatures:
                raise RuntimeError(
                    f"Generated duplicate mesh geometry for trajectory {sample}."
                )
            mesh_signatures.add(signature)
            triangles_sample = triangulate_vertices(
                vertices=vertices_sample,
                nx=args.nx,
                ny=args.ny,
                mesh_mode=args.mesh_mode,
                rng=rng,
            )
            centers_sample, areas_sample, quadrature_sample = triangle_geometry(
                vertices=vertices_sample, triangles=triangles_sample
            )
            if not np.isclose(
                np.sum(areas_sample), domain_area, rtol=1.0e-10, atol=1.0e-12
            ):
                raise ValueError(
                    f"Trajectory {sample} triangle areas do not sum to the domain area."
                )
            graph_sample = build_cell_graph(
                vertices=vertices_sample,
                triangles=triangles_sample,
                centers=centers_sample,
                areas=areas_sample,
            )

            (
                reference_vertices_sample,
                reference_triangles_sample,
                reference_parent_sample,
            ) = refine_triangular_mesh(
                vertices=vertices_sample,
                triangles=triangles_sample,
                levels=args.reference_refinement_levels,
            )
            (
                reference_centers_sample,
                reference_areas_sample,
                reference_quadrature_sample,
            ) = triangle_geometry(
                vertices=reference_vertices_sample,
                triangles=reference_triangles_sample,
            )
            reference_graph_sample = build_cell_graph(
                vertices=reference_vertices_sample,
                triangles=reference_triangles_sample,
                centers=reference_centers_sample,
                areas=reference_areas_sample,
            )

        vertices_list.append(vertices_sample)
        triangles_list.append(triangles_sample)
        centers_list.append(centers_sample)
        areas_list.append(areas_sample)
        quadrature_list.append(quadrature_sample)
        graphs.append(graph_sample)
        reference_vertices_list.append(reference_vertices_sample)
        reference_triangles_list.append(reference_triangles_sample)
        reference_centers_list.append(reference_centers_sample)
        reference_areas_list.append(reference_areas_sample)
        reference_quadrature_list.append(reference_quadrature_sample)
        reference_parents_list.append(reference_parent_sample)
        reference_graphs.append(reference_graph_sample)

    vertices = np.stack(vertices_list, axis=0)
    triangles = np.stack(triangles_list, axis=0)
    centers = np.stack(centers_list, axis=0)
    areas = np.stack(areas_list, axis=0)
    quadrature_points = np.stack(quadrature_list, axis=0)
    graph = _stack_mesh_graphs(graphs)
    reference_vertices = np.stack(reference_vertices_list, axis=0)
    reference_triangles = np.stack(reference_triangles_list, axis=0)
    reference_centers = np.stack(reference_centers_list, axis=0)
    reference_areas = np.stack(reference_areas_list, axis=0)
    reference_quadrature = np.stack(reference_quadrature_list, axis=0)
    reference_parent = np.stack(reference_parents_list, axis=0)
    reference_graph = _stack_mesh_graphs(reference_graphs)

    chemo, chemo_gradient, chemo_metadata = generate_chemoattractant_fields(
        num_samples=args.num_samples,
        quadrature_points=quadrature_points,
        lx=args.lx,
        ly=args.ly,
        num_sources_min=args.chemo_sources_min,
        num_sources_max=args.chemo_sources_max,
        amplitude_min=args.chemo_amplitude_min,
        amplitude_max=args.chemo_amplitude_max,
        sigma_min=args.chemo_sigma_min,
        sigma_max=args.chemo_sigma_max,
        baseline=args.chemo_baseline,
        margin_fraction=args.source_margin_fraction,
        shared_across_trajectories=args.shared_chemoattractant,
        rng=rng,
    )
    reference_density0, density_metadata = generate_initial_density(
        num_samples=args.num_samples,
        quadrature_points=reference_quadrature,
        chemo_metadata=chemo_metadata,
        lx=args.lx,
        ly=args.ly,
        num_blobs_min=args.density_blobs_min,
        num_blobs_max=args.density_blobs_max,
        amplitude_min=args.density_amplitude_min,
        amplitude_max=args.density_amplitude_max,
        sigma_min=args.density_sigma_min,
        sigma_max=args.density_sigma_max,
        background_min=args.density_background_min,
        background_max=args.density_background_max,
        margin_fraction=args.density_margin_fraction,
        primary_separation=args.primary_separation,
        rng=rng,
    )

    coarse_drift_speed = face_drift_velocity(
        face_midpoints=graph["shared_face_midpoints"],
        face_normals=graph["shared_face_normals"],
        chemo_metadata=chemo_metadata,
        chi=trajectory_chi,
    )
    reference_drift_speed = face_drift_velocity(
        face_midpoints=reference_graph["shared_face_midpoints"],
        face_normals=reference_graph["shared_face_normals"],
        chemo_metadata=chemo_metadata,
        chi=trajectory_chi,
    )
    coarse_stable_dt, coarse_trajectory_max_rates = stable_explicit_timestep(
        areas=areas,
        undirected_edges=graph["undirected_edge_index"],
        shared_face_lengths=graph["shared_face_lengths"],
        center_normal_distances=graph["center_normal_distances"],
        drift_speed=coarse_drift_speed,
        diffusion=args.diffusion,
        cfl=args.CFL,
    )
    reference_stable_dt, reference_trajectory_max_rates = stable_explicit_timestep(
        areas=reference_areas,
        undirected_edges=reference_graph["undirected_edge_index"],
        shared_face_lengths=reference_graph["shared_face_lengths"],
        center_normal_distances=reference_graph["center_normal_distances"],
        drift_speed=reference_drift_speed,
        diffusion=args.diffusion,
        cfl=args.reference_cfl,
    )
    if args.solver_dt is None:
        nominal_coarse_dt = coarse_stable_dt
    else:
        if args.solver_dt <= 0.0:
            raise ValueError("solver_dt must be positive.")
        if args.solver_dt > coarse_stable_dt * (1.0 + 1.0e-12):
            raise ValueError(
                f"Requested solver_dt={args.solver_dt:.8e} exceeds the "
                f"computed coarse-grid stable step {coarse_stable_dt:.8e}."
            )
        nominal_coarse_dt = float(args.solver_dt)
    frame_dt = nominal_coarse_dt * args.substeps_per_frame
    reference_substeps_per_frame = max(
        args.substeps_per_frame, int(np.ceil(frame_dt / reference_stable_dt - 1.0e-12)),
    )
    solver_dt = frame_dt / reference_substeps_per_frame

    rollout, interface_flux_transport = generate_rollouts(
        refined_density0=reference_density0,
        refined_areas=reference_areas,
        coarse_parent=reference_parent,
        coarse_areas=areas,
        coarse_undirected_edges=graph["undirected_edge_index"],
        undirected_edges=reference_graph["undirected_edge_index"],
        shared_face_lengths=reference_graph["shared_face_lengths"],
        centers=reference_centers,
        face_midpoints=reference_graph["shared_face_midpoints"],
        face_normals=reference_graph["shared_face_normals"],
        center_normal_distances=reference_graph["center_normal_distances"],
        drift_speed=reference_drift_speed,
        diffusion=args.diffusion,
        solver_dt=solver_dt,
        window=args.window,
        substeps_per_frame=reference_substeps_per_frame,
    )
    flux_implied_mass_change = np.zeros_like(rollout[:, 1:])
    for sample in range(args.num_samples):
        cell_i, cell_j = graph["undirected_edge_index"][sample]
        for frame in range(args.window):
            np.add.at(
                flux_implied_mass_change[sample, frame],
                cell_i,
                -interface_flux_transport[sample, frame],
            )
            np.add.at(
                flux_implied_mass_change[sample, frame],
                cell_j,
                interface_flux_transport[sample, frame],
            )
    observed_mass_change = (rollout[:, 1:] - rollout[:, :-1]) * areas[:, None, :]
    max_flux_balance_error = float(
        np.max(np.abs(flux_implied_mass_change - observed_mass_change))
    )
    if max_flux_balance_error > 1.0e-10:
        raise RuntimeError(
            "Aggregated reference face transports do not reproduce the stored "
            f"coarse-state update (max error {max_flux_balance_error:.8e})."
        )
    total_mass = np.sum(rollout * areas[:, None, :], axis=2)
    mass_error = total_mass - total_mass[:, [0]]
    max_mass_error = float(np.max(np.abs(mass_error)))
    minimum_density = float(np.min(rollout))
    if minimum_density < -1.0e-10:
        raise RuntimeError(
            f"Generated density became negative ({minimum_density:.8e}); "
            "reduce CFL or solver_dt."
        )
    drift_velocity = trajectory_chi[:, None, None] * chemo_gradient

    edge_attr_columns = (
        "delta_x,delta_y,distance,unit_x,unit_y,shared_face_length,"
        "face_normal_x,face_normal_y,center_normal_distance,transmissibility,"
        "source_cell_area,target_cell_area,"
        "source_is_boundary_cell"
    )
    metadata: Dict[str, object] = {
        "model": "prescribed_chemoattractant_drift_diffusion",
        "equation": "n_t + div(chi*n*grad(c) - D*grad(n)) = 0",
        "boundary": "no_flux",
        "state_representation": "triangle_cell_average",
        "chemoattractant_evolution": "prescribed_static_in_time",
        "chemoattractant_landscape": (
            "shared_across_trajectories"
            if args.shared_chemoattractant
            else "unique_per_trajectory"
        ),
        "num_unique_chemoattractant_landscapes": int(
            1 if args.shared_chemoattractant else args.num_samples
        ),
        "mesh_geometry": (
            "shared_across_trajectories_static_in_time"
            if args.shared_mesh
            else "unique_per_trajectory_static_in_time"
        ),
        "num_unique_meshes": int(1 if args.shared_mesh else args.num_samples),
        "mesh_mode": str(args.mesh_mode),
        "mesh_jitter": float(args.mesh_jitter),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "num_mesh_vertices": int(vertices.shape[1]),
        "num_cells": int(triangles.shape[1]),
        "num_undirected_edges": int(graph["undirected_edge_index"].shape[2]),
        "num_boundary_faces": int(graph["boundary_faces"].shape[1]),
        "lx": float(args.lx),
        "ly": float(args.ly),
        "domain_area": domain_area,
        "cell_area_min": float(np.min(areas)),
        "cell_area_mean": float(np.mean(areas)),
        "cell_area_max": float(np.max(areas)),
        "diffusion": float(args.diffusion),
        "chi": trajectory_chi,
        "chi_sampling": chi_sampling,
        "chi_shared_across_trajectories": bool(
            np.all(trajectory_chi == trajectory_chi[0])
        ),
        "num_unique_chi_values": int(np.unique(trajectory_chi).size),
        "chi_min": chi_distribution_min,
        "chi_max": chi_distribution_max,
        "chi_sample_min": float(np.min(trajectory_chi)),
        "chi_sample_max": float(np.max(trajectory_chi)),
        "CFL": float(args.CFL),
        "reference_CFL": float(args.reference_cfl),
        "coarse_solver_dt_stability_limit": float(coarse_stable_dt),
        "reference_solver_dt_stability_limit": float(reference_stable_dt),
        "solver_dt_stability_limit": float(reference_stable_dt),
        "nominal_coarse_solver_dt": float(nominal_coarse_dt),
        "solver_dt": float(solver_dt),
        "requested_substeps_per_frame": int(args.substeps_per_frame),
        "substeps_per_frame": int(reference_substeps_per_frame),
        "dt": float(frame_dt),
        "reference_refinement_levels": int(args.reference_refinement_levels),
        "reference_num_mesh_vertices": int(reference_vertices.shape[1]),
        "reference_num_cells": int(reference_triangles.shape[1]),
        "reference_spatial_scheme": (
            "barth_jespersen_muscl_face_normal_nonorthogonal_finite_volume"
        ),
        "reference_time_integrator": "SSP_RK2",
        "reference_projection": "sum_child_mass_divide_parent_area",
        "interface_flux_reference": (
            "integrated_refined_ssp_rk2_face_flux_aggregated_to_coarse_faces"
        ),
        "interface_flux_orientation": "undirected_edge_index_source_to_target",
        "interface_flux_transport_units": "mass_per_stored_frame",
        "interface_flux_rate_units": "mass_per_time",
        "max_abs_interface_flux_balance_error": max_flux_balance_error,
        "num_samples": int(args.num_samples),
        "window": int(args.window),
        "num_pairs": int(args.num_samples * args.window),
        "target_type": str(args.target_type),
        "x_columns": "cell_density,chemoattractant,drift_velocity_x,drift_velocity_y",
        "edge_attr_columns": edge_attr_columns,
        "seed": int(args.seed),
        "chemo_sources_min": int(args.chemo_sources_min),
        "chemo_sources_max": int(args.chemo_sources_max),
        "chemo_amplitude_min": float(args.chemo_amplitude_min),
        "chemo_amplitude_max": float(args.chemo_amplitude_max),
        "chemo_sigma_min": float(args.chemo_sigma_min),
        "chemo_sigma_max": float(args.chemo_sigma_max),
        "chemo_baseline": float(args.chemo_baseline),
        "source_margin_fraction": float(args.source_margin_fraction),
        "density_blobs_min": int(args.density_blobs_min),
        "density_blobs_max": int(args.density_blobs_max),
        "density_amplitude_min": float(args.density_amplitude_min),
        "density_amplitude_max": float(args.density_amplitude_max),
        "density_sigma_min": float(args.density_sigma_min),
        "density_sigma_max": float(args.density_sigma_max),
        "density_background_min": float(args.density_background_min),
        "density_background_max": float(args.density_background_max),
        "density_margin_fraction": float(args.density_margin_fraction),
        "primary_separation": float(args.primary_separation),
        "trajectory_max_outgoing_rate": reference_trajectory_max_rates,
        "coarse_trajectory_max_outgoing_rate": coarse_trajectory_max_rates,
        "density_min_observed": minimum_density,
        "density_max_observed": float(np.max(rollout)),
        "max_abs_mass_error": max_mass_error,
        **chemo_metadata,
        **density_metadata,
    }
    save_dataset(
        out_path=args.out,
        rollout=rollout,
        interface_flux_transport=interface_flux_transport,
        chemo=chemo,
        chemo_gradient=chemo_gradient,
        drift_velocity=drift_velocity,
        vertices=vertices,
        triangles=triangles,
        centers=centers,
        areas=areas,
        graph=graph,
        frame_dt=frame_dt,
        target_type=args.target_type,
        metadata=metadata,
    )

    print("Generated chemotactic cell-migration dataset")
    print(f"  out: {args.out}")
    print(f"  trajectories: {args.num_samples}")
    print(f"  rollout window: {args.window}")
    print(f"  total training pairs: {args.num_samples * args.window}")
    print(f"  mesh mode: {args.mesh_mode}")
    if args.shared_mesh:
        print("  trajectory meshes: 1 (shared across trajectories)")
    else:
        print(f"  unique trajectory meshes: {args.num_samples}")
    if args.shared_chemoattractant:
        print("  chemoattractant landscapes: 1 (shared across trajectories)")
    else:
        print(f"  chemoattractant landscapes: {args.num_samples}")
    print(f"  mesh vertices per trajectory: {vertices.shape[1]}")
    print(f"  coarse triangular cells / graph nodes: {triangles.shape[1]}")
    print(f"  reference triangular cells: {reference_triangles.shape[1]}")
    print(f"  undirected graph edges: {graph['undirected_edge_index'].shape[2]}")
    print(f"  boundary faces: {graph['boundary_faces'].shape[1]}")
    print(f"  cell area range: [{np.min(areas):.8e}, {np.max(areas):.8e}]")
    print(f"  diffusion D: {args.diffusion}")
    if chi_sampling == "fixed":
        print(f"  chemotactic sensitivity chi: {trajectory_chi[0]}")
    else:
        print(
            "  chemotactic sensitivity chi: sampled uniformly from "
            f"[{args.chi_range[0]}, {args.chi_range[1]}]"
        )
        print(
            "  sampled chi range: "
            f"[{np.min(trajectory_chi):.8g}, {np.max(trajectory_chi):.8g}]"
        )
    print(f"  reference solver_dt: {solver_dt:.8e}")
    print(f"  reference substeps per stored frame: {reference_substeps_per_frame}")
    print(f"  stored-frame dt: {frame_dt:.8e}")
    print(f"  x shape: {(args.num_samples * args.window, triangles.shape[1], 4)}")
    print(f"  rollout_states shape: {rollout[..., None].shape}")
    print(f"  density range: [{np.min(rollout):.8e}, {np.max(rollout):.8e}]")
    print(f"  max |mass(t)-mass(0)|: {max_mass_error:.8e}")
    print(f"  max interface-flux balance error: {max_flux_balance_error:.8e}")


if __name__ == "__main__":
    main()
