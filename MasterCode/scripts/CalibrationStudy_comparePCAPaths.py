"""Compare three PCA-to-motor calibration paths.

All paths use the same aligned radar/motor measurement and the same model-3
calibration segment.  They differ only in the data used to fit/apply PCA:

1. Full measurement -> PCA -> segment.
2. Align and split -> vstack segments -> PCA -> split.
3. Align and split -> PCA independently per segment.

The predictions are evaluated on segments 7 through 13.
"""

import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import detrend

from mrpro.operators.PCACompressionOp import PCACompressionOp


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RADAR = "Data/RadarTest/radar_20260729_145247.npz"
MOTOR = "Data/TimeLogs/time_log_20260729_145342.json"
OUTPUT_DIR = "Data/calibration_plots/PCA_path_comparison"

SEGMENT_LENGTH = 40.0
CALIBRATION_SEGMENT = 3
TEST_SEGMENTS = range(7, 14)
N_COMPONENTS = 510
MOTOR_SCALE = 2000.0
MOTOR_EPOCH: datetime.datetime | None = None


# -----------------------------------------------------------------------------
# Input and preprocessing
# -----------------------------------------------------------------------------
def load_radar(path: str):
    data = np.load(path, allow_pickle=True)
    return data["radar_cube"], data["timestamps"], data["time_cube"]


def load_motor_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as file:
        entries = json.load(file)["motor_log"]
    return (
        np.array([entry["t"] for entry in entries], dtype=float),
        np.array([entry["position"] for entry in entries], dtype=float),
    )


def radar_times_to_seconds(radar_datetimes) -> np.ndarray:
    start = radar_datetimes[0]
    return np.array([(timestamp - start).total_seconds() for timestamp in radar_datetimes])


def align_motor_to_radar(
    motor_times: np.ndarray,
    radar_datetimes,
    motor_epoch: datetime.datetime | None,
) -> np.ndarray:
    offset = 0.0 if motor_epoch is None else (
        motor_epoch - radar_datetimes[0]
    ).total_seconds()
    return motor_times + offset


def remove_clutter(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=0, keepdims=True)


def radar_features(radar_cube: np.ndarray) -> np.ndarray:
    """Return two clutter-removed channels as (frames, features)."""
    channel_1 = np.asarray(np.squeeze(radar_cube[0]), dtype=float)
    channel_2 = np.asarray(np.squeeze(radar_cube[1]), dtype=float)
    # channel_1 = detrend(remove_clutter(channel_1))
    # channel_2 = detrend(remove_clutter(channel_2))
    return np.hstack([channel_1, channel_2])


def align_and_split(
    radar_cube: np.ndarray,
    radar_times: np.ndarray,
    motor_times: np.ndarray,
    motor_positions: np.ndarray,
    segment_length: float,
) -> list[dict]:
    n_segments = int(np.floor(radar_times[-1] / segment_length))
    segments = []

    for segment_id in range(n_segments):
        start = segment_id * segment_length
        end = (segment_id + 1) * segment_length
        radar_mask = (radar_times >= start) & (radar_times < end)
        if np.count_nonzero(radar_mask) < 2:
            continue
        
        radar_segment = radar_cube[:, radar_mask, :]
        segment_radar_times = radar_times[radar_mask]
        motor_mask = (motor_times >= start) & (motor_times < end)

        segments.append({
            "id": segment_id,
            "features": radar_features(radar_segment),
            "radar_times": segment_radar_times,
            "motor_times": motor_times[motor_mask],
            "motor_positions": motor_positions[motor_mask],
        })

    return segments


# -----------------------------------------------------------------------------
# PCA and model 3
# -----------------------------------------------------------------------------
def first_pc(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = torch.as_tensor(data - mean, dtype=torch.float32)
    n_components = min(N_COMPONENTS, centered.shape[0], centered.shape[1])
    operator = PCACompressionOp(
        data=centered,
        n_components=n_components,
        centering=False,
    )
    return operator(centered)[0][:, 0].detach().cpu().numpy()


class MotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, values):
        return self.linear(values)


def train_model_3(pc1: np.ndarray, positions: np.ndarray) -> MotorNet:
    """Train the one-feature linear model used as model 3."""
    lower, upper = np.quantile(pc1, [0.1, 0.9])
    keep = (pc1 >= lower) & (pc1 <= upper)
    trimmed_pc1 = pc1[keep]
    trimmed_positions = positions[keep]

    changes = np.abs(np.diff(trimmed_positions, prepend=trimmed_positions[0]))
    moving = changes > 1e-4
    keep_static = np.arange(len(trimmed_positions)) % 30 == 0
    keep = moving | keep_static

    torch.manual_seed(0)
    model = MotorNet()
    inputs = torch.as_tensor(trimmed_pc1[keep, None], dtype=torch.float32)
    targets = torch.as_tensor(trimmed_positions[keep, None], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    loss_fn = nn.MSELoss()
    previous_loss = float("inf")
    patience_count = 0

    for _ in range(1000):
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if previous_loss - loss.item() < 1e-6:
            patience_count += 1
        else:
            patience_count = 0
        if patience_count >= 50:
            break
        previous_loss = loss.item()

    return model.eval()


def predict(model: MotorNet, pc1: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(pc1[:, None], dtype=torch.float32)).squeeze(1).numpy()


# -----------------------------------------------------------------------------
# Evaluation and plotting
# -----------------------------------------------------------------------------
def evaluate_path(
    model: MotorNet,
    path_name: str,
    path_pc1: dict[int, np.ndarray],
    segments: list[dict],
    ) -> list[dict]:
    results = []
    for segment in segments:
        segment_id = segment["id"]
        if segment_id not in TEST_SEGMENTS or len(segment["motor_times"]) == 0:
            continue

        predicted_radar = predict(model, path_pc1[segment_id])
        predicted_motor = np.interp(
            segment["motor_times"],
            segment["radar_times"],
            predicted_radar,
        )
        true_motor = segment["motor_positions"] / MOTOR_SCALE
        error = true_motor - predicted_motor
        results.append({
            "path": path_name,
            "segment": segment_id,
            "times": segment["motor_times"],
            "true": true_motor,
            "predicted": predicted_motor,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
        })
    return results


def plot_results(results: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_names = list(dict.fromkeys(result["path"] for result in results))
    figure, axes = plt.subplots(len(path_names), 1, figsize=(14, 4 * len(path_names)), sharex=True)
    axes = np.atleast_1d(axes)

    for axis, path_name in zip(axes, path_names):
        path_results = [result for result in results if result["path"] == path_name]
        for result in path_results:
            axis.plot(result["times"], result["true"], color="black", linewidth=2,
                      label="True motor" if result["segment"] == path_results[0]["segment"] else None)
            axis.plot(result["times"], result["predicted"], linewidth=1.5,
                      label=f"Prediction, segment {result['segment']}")
        mean_rmse = np.mean([result["rmse"] for result in path_results])
        mean_mae = np.mean([result["mae"] for result in path_results])
        axis.set_title(f"{path_name} | mean RMSE={mean_rmse:.4f}, mean MAE={mean_mae:.4f}")
        axis.set_ylabel("Position / 2000")
        axis.grid(alpha=0.3)
        axis.legend(ncol=2, fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, "paths_segments_7_13.png"), dpi=150)
    plt.close(figure)


def print_metrics(results: list[dict]) -> None:
    for path_name in dict.fromkeys(result["path"] for result in results):
        path_results = [result for result in results if result["path"] == path_name]
        print(
            f"{path_name}: "
            f"RMSE={np.mean([r['rmse'] for r in path_results]):.6f}, "
            f"MAE={np.mean([r['mae'] for r in path_results]):.6f}"
        )


def main() -> None:
    radar_cube, _, time_cube = load_radar(RADAR)
    rx1 = radar_cube[0, :, :]
    rx2 = radar_cube[1, :, :]
    rx1_cf = remove_clutter(rx1)
    rx2_cf = remove_clutter(rx2)
    rx1_cf = detrend(rx1_cf)
    rx2_cf = detrend(rx2_cf)
    radar_cube[0, :, :] = rx1_cf
    radar_cube[1, :, :] = rx2_cf

    raw_motor_times, motor_positions = load_motor_log(MOTOR)
    radar_times = radar_times_to_seconds(time_cube[0, :])
    motor_times = align_motor_to_radar(raw_motor_times, time_cube[0, :], MOTOR_EPOCH)

    segments = align_and_split(
        radar_cube, radar_times, motor_times, motor_positions, SEGMENT_LENGTH
    )
    usable_segments = [segment for segment in segments if len(segment["motor_times"]) >= 2]
    calibration = next(segment for segment in usable_segments if segment["id"] == CALIBRATION_SEGMENT)

    reference_segment = next(segment for segment in usable_segments if segment["id"] == 0)
    reference_mean = np.mean(reference_segment["features"], axis=0)

    # Path 1: PCA is fitted on every radar frame, then segmented.
    full_features = radar_features(radar_cube)
    full_pc1 = first_pc(full_features, reference_mean)
    full_pc1_by_segment = {}
    for segment in usable_segments:
        start = segment["id"] * SEGMENT_LENGTH
        mask = (radar_times >= start) & (radar_times < start + SEGMENT_LENGTH)
        full_pc1_by_segment[segment["id"]] = full_pc1[mask]



    # Path 2: the aligned/split features are vstacked before one global PCA.
    stacked_features = np.vstack([segment["features"] for segment in usable_segments])
    stacked_pc1 = first_pc(stacked_features, reference_mean)
    stacked_pc1_by_segment = {}
    cursor = 0
    for segment in usable_segments:
        length = len(segment["features"])
        stacked_pc1_by_segment[segment["id"]] = stacked_pc1[cursor:cursor + length]
        cursor += length



    # Path 3: fit/apply PCA independently to each segment.
    segment_pc1_by_segment = {
        segment["id"]: first_pc(segment["features"], reference_mean)
        for segment in usable_segments
    }

    path_definitions = {
        "Full measurement -> PCA -> segment": full_pc1_by_segment,
        "Vstack segments -> global PCA": stacked_pc1_by_segment,
        "Segment 3 PCA -> model 3": segment_pc1_by_segment,
    }

    all_results = []
    for path_name, pc1_by_segment in path_definitions.items():
        model_3 = train_model_3(
            pc1_by_segment[CALIBRATION_SEGMENT],
            np.interp(
                calibration["radar_times"],
                calibration["motor_times"],
                calibration["motor_positions"],
            ) / MOTOR_SCALE,
        )
        all_results.extend(evaluate_path(model_3, path_name, pc1_by_segment, usable_segments))

    plot_results(all_results)
    print_metrics(all_results)
    print(f"Saved comparison plot to {OUTPUT_DIR}/paths_segments_7_13.png")


if __name__ == "__main__":
    main()