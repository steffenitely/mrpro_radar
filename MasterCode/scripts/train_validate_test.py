"""Train and evaluate a radar-to-motor-position regression model.

Example
-------
python -m MasterCode.scripts.train_validate_test \\
    --pair radar_run_1.npz motor_run_1.json \\
    --pair radar_run_2.npz motor_run_2.json \\
    --output-dir results/run_01

Radar timestamps are converted to seconds from the first radar frame. Motor log
`t` values are aligned to that time origin. Without `--motor-epochs`, each
motor log is assumed to start at the same time as its radar recording.
Radar files containing pickled Python datetime objects must be trusted inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class PreparedSegment:
    """Features, aligned targets, and timestamps for one measurement segment."""

    segment_id: int
    features: np.ndarray
    targets: np.ndarray
    radar_times: np.ndarray
    motor_times: np.ndarray
    motor_positions: np.ndarray


class MotorNetRawData(nn.Module):
    """MLP mapping concatenated radar channels to scaled motor position."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        hidden_1 = max(128, input_size // 2)
        hidden_2 = max(64, input_size // 4)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def load_motor_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate elapsed-time and position arrays from a motor JSON log."""
    with path.open(encoding="utf-8") as file:
        document = json.load(file)

    try:
        entries = document["motor_log"]
        times = np.asarray([entry["t"] for entry in entries], dtype=np.float64)
        positions = np.asarray(
            [entry["position"] for entry in entries],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid motor log format: {path}") from error

    if times.size < 2:
        raise ValueError(f"Motor log must contain at least two samples: {path}")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError(f"Motor log contains non-finite values: {path}")

    order = np.argsort(times)
    times = times[order]
    positions = positions[order]
    times, unique_indices = np.unique(times, return_index=True)
    positions = positions[unique_indices]

    if times.size < 2:
        raise ValueError(f"Motor log needs at least two distinct times: {path}")

    return times, positions


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").astype(datetime)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Unsupported radar timestamp value: {value!r}")


def load_radar_record(path: Path) -> tuple[np.ndarray, np.ndarray, datetime]:
    """Load radar samples and frame times from the current or legacy NPZ format."""
    try:
        with np.load(path, allow_pickle=True) as archive:
            radar_cube = np.asarray(archive["radar_cube"], dtype=np.float32)

            if "time_cube" in archive.files:
                time_cube = np.asarray(archive["time_cube"], dtype=object)
                radar_timestamps = time_cube[0] if time_cube.ndim == 2 else time_cube
            elif "timestamps" in archive.files:
                radar_timestamps = np.asarray(archive["timestamps"], dtype=object)
            else:
                raise ValueError("NPZ must contain 'time_cube' or 'timestamps'")
    except (OSError, KeyError) as error:
        raise ValueError(f"Could not load radar recording: {path}") from error

    if radar_cube.ndim != 3 or radar_cube.shape[0] < 2:
        raise ValueError(
            f"Expected radar_cube with shape (2, frames, samples), got {radar_cube.shape}"
        )
    if radar_cube.shape[1] != len(radar_timestamps):
        raise ValueError(
            "Radar frame count does not match timestamp count: "
            f"{radar_cube.shape[1]} frames versus {len(radar_timestamps)} timestamps"
        )
    if len(radar_timestamps) < 2:
        raise ValueError(f"Radar recording must contain at least two frames: {path}")

    datetime_values = [_as_datetime(value) for value in radar_timestamps]
    radar_start = datetime_values[0]
    radar_times = np.asarray(
        [(value - radar_start).total_seconds() for value in datetime_values],
        dtype=np.float64,
    )

    if np.any(np.diff(radar_times) < 0):
        raise ValueError(f"Radar timestamps are not ordered: {path}")

    return radar_cube, radar_times, radar_start


def prepare_recording(
    radar_path: Path,
    motor_path: Path,
    segment_length: float,
    position_scale: float,
    first_segment_id: int,
    motor_epoch: datetime | None = None,
) -> tuple[list[PreparedSegment], int]:
    """Align one radar/motor pair and split it into fixed-duration segments."""
    radar_cube, radar_times, radar_start = load_radar_record(radar_path)
    motor_times, motor_positions = load_motor_log(motor_path)

    if motor_epoch is None:
        motor_offset = 0.0
    else:
        motor_offset = (motor_epoch - radar_start).total_seconds()
    aligned_motor_times = motor_times + motor_offset

    segment_count = int(np.floor(radar_times[-1] / segment_length))
    prepared: list[PreparedSegment] = []

    for local_segment_id in range(segment_count):
        segment_id = first_segment_id + local_segment_id
        start = local_segment_id * segment_length
        end = start + segment_length

        radar_mask = (radar_times >= start) & (radar_times < end)
        motor_mask = (aligned_motor_times >= start) & (aligned_motor_times < end)
        segment_radar_times = radar_times[radar_mask]
        segment_motor_times = aligned_motor_times[motor_mask]
        segment_motor_positions = motor_positions[motor_mask]

        if segment_radar_times.size < 2 or segment_motor_times.size < 2:
            continue

        features = np.concatenate(
            (radar_cube[0, radar_mask], radar_cube[1, radar_mask]),
            axis=1,
        )
        interpolated_positions = np.interp(
            segment_radar_times,
            aligned_motor_times,
            motor_positions,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(interpolated_positions)

        if np.count_nonzero(valid) < 2:
            continue

        prepared.append(
            PreparedSegment(
                segment_id=segment_id,
                features=features[valid],
                targets=interpolated_positions[valid] / position_scale,
                radar_times=segment_radar_times[valid],
                motor_times=segment_motor_times,
                motor_positions=segment_motor_positions,
            )
        )

    return prepared, first_segment_id + segment_count


def stack_segments(
    segments: list[PreparedSegment],
    requested_ids: list[int],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [segment for segment in segments if segment.segment_id in requested_ids]
    found_ids = {segment.segment_id for segment in selected}
    missing_ids = sorted(set(requested_ids) - found_ids)

    if missing_ids:
        raise ValueError(f"No usable data for {split_name} segment(s): {missing_ids}")

    features = np.concatenate([segment.features for segment in selected], axis=0)
    targets = np.concatenate([segment.targets for segment in selected]).reshape(-1, 1)
    return features, targets


def train_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    epochs: int,
    patience: int,
    learning_rate: float,
    wandb_run: Any | None = None,
) -> tuple[MotorNetRawData, dict[str, float | int]]:
    """Train with validation-loss early stopping and return the best model."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotorNetRawData(train_features.shape[1]).to(device)

    x_train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    x_validation = torch.as_tensor(
        validation_features,
        dtype=torch.float32,
        device=device,
    )
    y_validation = torch.as_tensor(
        validation_targets,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_function(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(x_validation), y_validation).item()

        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if wandb_run is not None and epoch % 10 == 0:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss.item(),
                    "validation_loss": validation_loss,
                }
            )

        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint")

    model.load_state_dict(best_state)
    model.cpu()
    model.eval()

    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss}


def evaluate_segments(
    model: MotorNetRawData,
    segments: list[PreparedSegment],
    output_dir: Path,
    split_name: str,
    position_scale: float,
    wandb_run: Any | None = None,
) -> list[dict[str, float | int]]:
    """Evaluate a split, save per-segment plots, and return error metrics."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | int]] = []

    for segment in segments:
        model_input = torch.as_tensor(segment.features, dtype=torch.float32)
        with torch.no_grad():
            predictions = model(model_input).squeeze(-1).numpy()

        predicted_at_motor_times = np.interp(
            segment.motor_times,
            segment.radar_times,
            predictions,
            left=np.nan,
            right=np.nan,
        )
        measured_positions = segment.motor_positions / position_scale
        valid = np.isfinite(predicted_at_motor_times)

        if np.count_nonzero(valid) < 2:
            continue

        errors = measured_positions[valid] - predicted_at_motor_times[valid]
        metrics = {
            "segment": segment.segment_id,
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "max_absolute_error": float(np.max(np.abs(errors))),
        }
        results.append(metrics)

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(
            segment.motor_times[valid],
            measured_positions[valid],
            label="Measured motor position",
            linewidth=2,
        )
        axis.plot(
            segment.motor_times[valid],
            predicted_at_motor_times[valid],
            label="Radar prediction",
            alpha=0.8,
        )
        axis.set_xlabel("Time from radar start (s)")
        axis.set_ylabel(f"Scaled position (position / {position_scale:g})")
        axis.set_title(
            f"{split_name} segment {segment.segment_id}: "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
        )
        axis.grid(True)
        axis.legend()
        figure.tight_layout()

        plot_path = output_dir / f"{split_name.lower()}_segment_{segment.segment_id}.png"
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{split_name.lower()}/segment_{segment.segment_id}/rmse": metrics["rmse"],
                    f"{split_name.lower()}/segment_{segment.segment_id}/mae": metrics["mae"],
                }
            )

        plt.close(figure)

    if not results:
        raise ValueError(f"No evaluable segments found for {split_name}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an MLP using paired radar and motor recordings."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("RADAR_NPZ", "MOTOR_JSON"),
        required=True,
        help="Radar NPZ and matching motor JSON; repeat for each recording.",
    )
    parser.add_argument(
        "--motor-epochs",
        nargs="*",
        help=(
            "Optional ISO datetimes corresponding to each pair's motor-log t=0. "
            "Omit to assume each motor log and radar recording start together."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/train_validate_test"))
    parser.add_argument("--segment-length", type=float, default=40.0)
    parser.add_argument("--position-scale", type=float, default=2000.0)
    parser.add_argument("--train-segments", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--validation-segments", type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument(
        "--test-segments",
        type=int,
        nargs="+",
        default=list(range(8, 21)),
    )
    parser.add_argument("--reference-segment", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--wandb-project", help="Opt in to Weights & Biases logging.")

    args = parser.parse_args()
    if args.segment_length <= 0 or args.position_scale <= 0:
        parser.error("--segment-length and --position-scale must be positive")
    if args.epochs <= 0 or args.patience <= 0 or args.learning_rate <= 0:
        parser.error("epochs, patience, and learning rate must be positive")
    if args.motor_epochs and len(args.motor_epochs) != len(args.pair):
        parser.error("Provide one --motor-epochs value per --pair")

    split_ids = [
        set(args.train_segments),
        set(args.validation_segments),
        set(args.test_segments),
    ]
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        parser.error("Train, validation, and test segment IDs must not overlap")
    if args.reference_segment in set.union(*split_ids):
        parser.error("The reference segment must not be included in a data split")

    return args


def main() -> None:
    args = parse_args()
    motor_epochs = [
        datetime.fromisoformat(value) for value in (args.motor_epochs or [])
    ]

    all_segments: list[PreparedSegment] = []
    next_segment_id = 0
    for index, (radar_name, motor_name) in enumerate(args.pair):
        epoch = motor_epochs[index] if motor_epochs else None
        segments, next_segment_id = prepare_recording(
            radar_path=Path(radar_name),
            motor_path=Path(motor_name),
            segment_length=args.segment_length,
            position_scale=args.position_scale,
            first_segment_id=next_segment_id,
            motor_epoch=epoch,
        )
        all_segments.extend(segments)

    if args.reference_segment not in {segment.segment_id for segment in all_segments}:
        raise ValueError(f"Reference segment {args.reference_segment} has no usable data")

    training_segments = [
        segment for segment in all_segments if segment.segment_id in args.train_segments
    ]
    validation_segments = [
        segment
        for segment in all_segments
        if segment.segment_id in args.validation_segments
    ]
    test_segments = [
        segment for segment in all_segments if segment.segment_id in args.test_segments
    ]

    train_features, train_targets = stack_segments(
        training_segments,
        args.train_segments,
        "training",
    )
    validation_features, validation_targets = stack_segments(
        validation_segments,
        args.validation_segments,
        "validation",
    )

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "Install wandb or omit --wandb-project to disable tracking"
            ) from error
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    try:
        model, training_summary = train_model(
            train_features=train_features,
            train_targets=train_targets,
            validation_features=validation_features,
            validation_targets=validation_targets,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            wandb_run=wandb_run,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.output_dir / "motor_net_raw_data.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": train_features.shape[1],
                "position_scale": args.position_scale,
                "reference_segment": args.reference_segment,
            },
            model_path,
        )

        validation_results = evaluate_segments(
            model,
            validation_segments,
            args.output_dir / "validation",
            "Validation",
            args.position_scale,
            wandb_run,
        )
        test_results = evaluate_segments(
            model,
            test_segments,
            args.output_dir / "test",
            "Test",
            args.position_scale,
            wandb_run,
        )

        results = {
            "training": training_summary,
            "validation": validation_results,
            "test": test_results,
            "mean_test_rmse": float(np.mean([item["rmse"] for item in test_results])),
            "mean_test_mae": float(np.mean([item["mae"] for item in test_results])),
            "mean_test_max_absolute_error": float(
                np.mean([item["max_absolute_error"] for item in test_results])
            ),
            "model_path": str(model_path),
            "segment_ids": {
                "reference": args.reference_segment,
                "train": args.train_segments,
                "validation": args.validation_segments,
                "test": args.test_segments,
            },
        }
        results_path = args.output_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

        if wandb_run is not None:
            wandb_run.log(
                {
                    "final_test/mean_rmse": results["mean_test_rmse"],
                    "final_test/mean_mae": results["mean_test_mae"],
                    "final_test/mean_max_absolute_error": results[
                        "mean_test_max_absolute_error"
                    ],
                }
            )
            wandb_run.save(str(model_path))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
