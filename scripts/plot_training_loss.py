#!/usr/bin/env python3
"""Plot chemotaxis GNN training and validation histories.

The checkpoint written by ``scripts/train.py`` contains the complete epoch
history. This script plots normalized autoregressive node MSE, physical
density RMSE, and cell-area-weighted state relative L2 when available. The
best-epoch marker follows the validation metric selected during training.

Example:

    python scripts/plot_training_loss.py \
        --checkpoint runs/chemotaxis/sageconv_delta_ar1/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_mpl_cache = Path(tempfile.gettempdir()) / "chemotaxis-matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires PyTorch to read checkpoint.pt."
    ) from exc


def load_history(
    checkpoint_path: Path,
) -> Tuple[Dict[str, object], List[Mapping[str, object]]]:
    """Load and validate a trusted training checkpoint and its history."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain a dictionary.")
    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Checkpoint history is missing or empty.")
    required = {
        "epoch",
        "train_normalized_mse",
        "validation_normalized_mse",
    }
    for index, record in enumerate(history):
        if not isinstance(record, Mapping):
            raise ValueError(f"History record {index} is not a dictionary.")
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"History record {index} is missing fields: {sorted(missing)}"
            )
    return checkpoint, history


def _history_array(
    history: Sequence[Mapping[str, object]], name: str
) -> np.ndarray:
    values = np.asarray([record[name] for record in history], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Training history field {name!r} is non-finite.")
    return values


def plot_history(
    checkpoint: Dict[str, object],
    history: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    yscale: str,
    title: str,
    dpi: int,
) -> None:
    """Save normalized loss and physical density-error curves."""
    epochs = _history_array(history, "epoch").astype(np.int64)
    train_loss = _history_array(history, "train_normalized_mse")
    validation_loss = _history_array(history, "validation_normalized_mse")
    has_state_rmse = all(
        "train_next_state_rmse" in record
        and "validation_next_state_rmse" in record
        for record in history
    )
    has_state_rel_l2 = all(
        "train_rollout_state_rel_l2" in record
        and "validation_rollout_state_rel_l2" in record
        for record in history
    )
    all_plotted = [train_loss, validation_loss]
    if has_state_rmse:
        train_state_rmse = _history_array(history, "train_next_state_rmse")
        validation_state_rmse = _history_array(
            history, "validation_next_state_rmse"
        )
        all_plotted.extend((train_state_rmse, validation_state_rmse))
    if has_state_rel_l2:
        train_state_rel_l2 = _history_array(history, "train_rollout_state_rel_l2")
        validation_state_rel_l2 = _history_array(
            history, "validation_rollout_state_rel_l2"
        )
        all_plotted.extend((train_state_rel_l2, validation_state_rel_l2))
    if yscale == "log" and any(np.any(values <= 0.0) for values in all_plotted):
        raise ValueError("A logarithmic axis requires strictly positive values.")

    selection_metric, best_index = best_history_index(checkpoint, history)
    best_epoch = int(epochs[best_index])

    columns = 1 + int(has_state_rmse) + int(has_state_rel_l2)
    figure, axes = plt.subplots(
        1, columns, figsize=(7.5 * columns, 5.1), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)

    training_args = checkpoint.get("training_args", {})
    if not isinstance(training_args, Mapping):
        training_args = {}
    node_loss_normalization = str(
        checkpoint.get(
            "node_loss_normalization",
            training_args.get("node_loss_normalization", "density"),
        )
    )
    normalized_loss_ylabel = (
        "Normalized target-space MSE"
        if node_loss_normalization == "target"
        else "Normalized state-space MSE"
    )
    panels = [
        (
            axes_array[0],
            train_loss,
            validation_loss,
            "Normalized autoregressive loss",
            normalized_loss_ylabel,
        )
    ]
    if has_state_rmse:
        panels.append(
            (
                axes_array[1],
                train_state_rmse,
                validation_state_rmse,
                "Physical prediction error",
                "Cell-density RMSE",
            )
        )
    if has_state_rel_l2:
        panels.append(
            (
                axes_array[len(panels)],
                train_state_rel_l2,
                validation_state_rel_l2,
                "Physical relative prediction error",
                r"Cell-area-weighted state relative $L_2$",
            )
        )

    for axis, train_values, validation_values, panel_title, ylabel in panels:
        axis.plot(epochs, train_values, color="tab:blue", label="Train")
        axis.plot(
            epochs, validation_values, color="tab:orange", label="Validation"
        )
        axis.axvline(
            best_epoch,
            color="0.35",
            linestyle="--",
            linewidth=1.1,
            label=f"Best epoch ({best_epoch})",
        )
        axis.scatter(
            [best_epoch],
            [validation_values[best_index]],
            color="tab:orange",
            edgecolor="white",
            linewidth=0.8,
            s=45,
            zorder=3,
        )
        axis.set_title(panel_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_yscale(yscale)
        axis.grid(alpha=0.28)
        axis.legend()

    model_name = str(checkpoint.get("model_name", checkpoint.get("model_class", "GNN")))
    target_type = str(checkpoint.get("target_type", "unknown"))
    rollout_steps = checkpoint.get("autoregressive_steps", "?")
    figure.suptitle(
        f"{title}\n{model_name}; target={target_type}; "
        f"training rollout={rollout_steps} step(s); selection={selection_metric}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def best_history_index(
    checkpoint: Mapping[str, object], history: Sequence[Mapping[str, object]]
) -> Tuple[str, int]:
    """Honor saved selection metadata, falling back to nMSE for old runs."""
    training_args = checkpoint.get("training_args", {})
    if not isinstance(training_args, Mapping):
        training_args = {}
    metric = str(checkpoint.get(
        "early_stopping_metric",
        training_args.get("early_stopping_metric", "normalized_mse"),
    ))
    values = _history_array(history, f"validation_{metric}")
    best_index = int(np.argmin(values))
    saved_epoch = checkpoint.get("best_epoch")
    if saved_epoch is not None:
        matching = [
            index for index, record in enumerate(history)
            if int(record["epoch"]) == int(saved_epoch)
        ]
        if matching:
            best_index = matching[0]
    return metric, best_index


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation histories from checkpoint.pt."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to loss_vs_epoch.png beside the checkpoint.",
    )
    parser.add_argument("--yscale", choices=("log", "linear"), default="log")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "loss_vs_epoch.png"
    )
    title = args.title or f"Training history: {checkpoint_path.parent.name}"
    checkpoint, history = load_history(checkpoint_path)
    plot_history(
        checkpoint,
        history,
        output_path,
        yscale=args.yscale,
        title=title,
        dpi=args.dpi,
    )
    selection_metric, best_index = best_history_index(checkpoint, history)
    selected_value = float(history[best_index][f"validation_{selection_metric}"])
    print(
        f"best validation epoch: {int(history[best_index]['epoch'])} "
        f"({selection_metric}={selected_value:.6e})"
    )
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()
