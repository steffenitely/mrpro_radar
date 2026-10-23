"""Calibrate radar PCA against motor position with selectable means and PCA.

Set the paths and options below, then run this file. If reference paths are set,
the reference radar is used only to calculate the centering mean. PCA and motor
regression models are always fitted from selected segments of the active
recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import detrend
from sklearn.linear_model import LinearRegression

from mrpro.operators.PCACompressionOp import PCACompressionOp


# Active recording to evaluate.
RADAR_PATH = Path("Data/Input/CalibrationTrial/RadarLogs/radar_20261022_131201.npz")
MOTOR_PATH = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_131300.json")

# Set both paths to use a separate calibration recording. Leave both as None
# unless MEAN_SOURCE is "reference"; then only the radar path is needed.
REFERENCE_RADAR_PATH: Path | None = Path("Data/Input/CalibrationTrial/RadarLogs/radar_20261022_143214.npz")

# Choose the feature mean used before PCA:
# - "global": mean of the full active recording
# - "segmentwise": a separate mean for each calibration segment
# - "reference": mean of the full reference radar recording
MEAN_SOURCE: Literal["global", "segmentwise", "reference"] = "reference"

# Choose one PCA fitted from the selected calibration segments, or a separate
# PCA fitted from each selected calibration segment. The reference recording
# is never used to fit a PCA or a regression model.
PCA_MODE: Literal["global", "per_calibration_segment"] = "global"
SEGMENTATION_MODE: Literal["fixed_duration", "phase_notes"] = "phase_notes"
SEGMENT_LENGTH_SECONDS = 40.0
N_PCA_COMPONENTS = 510
MOTOR_POSITION_SCALE = 2000.0

# Choose which segments each get their own PCA and regression model.
# Use None for every segment. Example: (0, 2, 5).
CALIBRATION_SEGMENT_INDICES: tuple[int, ...] | None = (0, 1, 2, 3, 4, 5) 
# Compare model fitting with different retained portions of each segment's PCA
# component range. For example, 80 keeps the center 80% and trims 10% per end.
CALIBRATION_PCA_SPAN_PERCENTAGES: tuple[float, ...] = (60.0, 100.0, 90.0, 80.0, 70.0, 50.0)

CHOSEN_NAME = "mid_cal"
OUTPUT_DIR = Path("Data/Output/CalibrationTrial") / CHOSEN_NAME
RESULTS_DIR = OUTPUT_DIR / "calibration_matrix"
PREDICTION_PLOTS_DIR = OUTPUT_DIR / "segment_predictions"
REGRESSION_MODELS_DIR = OUTPUT_DIR / "regression_models"
MODEL_PERFORMANCE_DIR = OUTPUT_DIR / "model_performance"


@dataclass
class Recording:
    """Radar features and motor positions on their respective time axes."""

    radar_times: np.ndarray
    features: np.ndarray
    motor_times: np.ndarray
    motor_positions: np.ndarray
    phase_notes: list[tuple[float, str]]


@dataclass
class Segment:
    """One fixed-duration part of a recording."""

    index: int
    label: str
    radar_times: np.ndarray
    features: np.ndarray
    motor_times: np.ndarray
    motor_positions: np.ndarray


@dataclass
class Calibration:
    """PCA and motor-position model fitted to one calibration segment."""

    segment_index: int
    segment_label: str
    feature_mean: np.ndarray
    pca: PCACompressionOp
    motor_model: LinearRegression
    fit_components: np.ndarray
    fit_positions: np.ndarray
    pca_span_percent: float


@dataclass
class Evaluation:
    """Error matrices and radar-time predictions for every calibration/test pair."""

    rmse: np.ndarray
    mae: np.ndarray
    max_error: np.ndarray
    predictions: list[list[np.ndarray]]


def load_motor_log(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, str]]]:
    """Load motor samples and phase-finished notes from a motor JSON log."""
    with path.open(encoding="utf-8") as file:
        entries = json.load(file)["motor_log"]

    motor_samples = []
    phase_notes = []
    for index, entry in enumerate(entries):
        if "position" in entry:
            motor_samples.append((entry["t"], entry["position"]))

        note = entry.get("note")
        if note is not None and note.startswith("Phase finished:"):
            phase_name = note.removeprefix("Phase finished:").strip()
            phase_notes.append((entry["t"], phase_name))

        if "position" not in entry and note is None:
            raise ValueError(f"Motor log entry {index} has neither a position nor a note.")

    times = np.array([sample[0] for sample in motor_samples], dtype=float)
    positions = np.array([sample[1] for sample in motor_samples], dtype=float)
    return times, positions, phase_notes


def extract_radar_features(radar_cube: np.ndarray) -> np.ndarray:
    """Remove the per-range-bin mean and combine the two radar channels."""
    channels = []
    for channel_index in range(2):
        channel = np.asarray(radar_cube[channel_index], dtype=float)
        channel_without_clutter = channel - np.mean(channel, axis=0, keepdims=True)
        channels.append(detrend(channel_without_clutter, axis=-1))
    return np.hstack(channels)


def load_recording(radar_path: Path, motor_path: Path) -> Recording:
    """Load radar and motor data on their shared experiment-relative timebase."""
    with np.load(radar_path, allow_pickle=True) as radar_file:
        radar_cube = radar_file["radar_cube"]
        radar_times = np.asarray(radar_file["time_cube"][0, :], dtype=float)

    motor_times, motor_positions, phase_notes = load_motor_log(motor_path)

    return Recording(
        radar_times=radar_times,
        features=extract_radar_features(radar_cube),
        motor_times=motor_times,
        motor_positions=motor_positions,
        phase_notes=phase_notes,
    )


def load_radar_features(radar_path: Path) -> np.ndarray:
    """Load radar data and return processed features without using motor data."""
    with np.load(radar_path, allow_pickle=True) as radar_file:
        radar_cube = radar_file["radar_cube"]
    return extract_radar_features(radar_cube)


def split_into_segments(recording: Recording) -> list[Segment]:
    """Split a recording by fixed duration or by phase-finished notes."""
    if SEGMENTATION_MODE == "phase_notes":
        segment_start = recording.radar_times[0]
        recording_end = recording.radar_times[-1]
        intervals = []
        last_phase_name = None

        for segment_end, phase_name in recording.phase_notes:
            if segment_end <= segment_start or segment_end > recording_end:
                continue

            intervals.append((phase_name, segment_start, segment_end, False))
            segment_start = segment_end
            last_phase_name = phase_name

        # The interval after the final note is also a segment. Include its last
        # radar frame so the note boundaries cover the complete recording.
        if segment_start < recording_end:
            label = f"after_{last_phase_name}" if last_phase_name else "final_segment"
            intervals.append((label, segment_start, recording_end, True))

        segments = []
        for label, start, end, include_end in intervals:
            segment = make_segment(
                recording,
                len(segments),
                label,
                start,
                end,
                include_end=include_end,
            )
            if segment is not None:
                segments.append(segment)
        return segments

    n_segments = int(np.floor(recording.radar_times[-1] / SEGMENT_LENGTH_SECONDS))
    segments = []

    for index in range(n_segments):
        start = index * SEGMENT_LENGTH_SECONDS
        end = (index + 1) * SEGMENT_LENGTH_SECONDS
        segment = make_segment(recording, index, f"segment_{index}", start, end)
        if segment is not None:
            segments.append(segment)

    return segments


def make_segment(
    recording: Recording,
    index: int,
    label: str,
    start: float,
    end: float,
    include_end: bool = False,
) -> Segment | None:
    """Extract one interval, omitting intervals without enough samples."""
    if include_end:
        radar_mask = (recording.radar_times >= start) & (recording.radar_times <= end)
        motor_mask = (recording.motor_times >= start) & (recording.motor_times <= end)
    else:
        radar_mask = (recording.radar_times >= start) & (recording.radar_times < end)
        motor_mask = (recording.motor_times >= start) & (recording.motor_times < end)

    if np.count_nonzero(radar_mask) < 2 or np.count_nonzero(motor_mask) < 2:
        return None

    return Segment(
        index=index,
        label=label,
        radar_times=recording.radar_times[radar_mask],
        features=recording.features[radar_mask],
        motor_times=recording.motor_times[motor_mask],
        motor_positions=recording.motor_positions[motor_mask],
    )


def make_pca(data: np.ndarray) -> PCACompressionOp:
    """Fit the PCA projection to already-centered feature rows."""
    tensor_data = torch.as_tensor(data, dtype=torch.float32)
    return PCACompressionOp(
        data=tensor_data,
        n_components=N_PCA_COMPONENTS,
        centering=False,
    )


def apply_pca(pca: PCACompressionOp, centered_features: np.ndarray) -> np.ndarray:
    """Project centered feature rows and return the first principal component."""
    tensor_data = torch.as_tensor(centered_features, dtype=torch.float32)
    projected = pca(tensor_data)[0]
    return projected[:, 0].detach().cpu().numpy()


def fit_motor_model(
    segment: Segment,
    pca: PCACompressionOp,
    feature_mean: np.ndarray,
    pca_span_percent: float,
) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    """Fit using a centered percentage of the PCA range, not motor positions."""
    if not 0.0 < pca_span_percent <= 100.0:
        raise ValueError("Every retained PCA-span percentage must be in (0, 100].")

    component = apply_pca(pca, segment.features - feature_mean)
    motor_on_radar = np.interp(
        segment.radar_times,
        segment.motor_times,
        segment.motor_positions,
    )
    target = motor_on_radar / MOTOR_POSITION_SCALE

    pca_min = component.min()
    pca_span = np.ptp(component)
    trim_each_side = (100.0 - pca_span_percent) / 200.0
    lower = pca_min + trim_each_side * pca_span
    upper = pca_min + (1.0 - trim_each_side) * pca_span
    fit_mask = (component >= lower) & (component <= upper)

    model = LinearRegression()
    fit_components = component[fit_mask]
    fit_positions = target[fit_mask]
    model.fit(fit_components.reshape(-1, 1), fit_positions)
    return model, fit_components, fit_positions


def fit_calibrations(
    segments: list[Segment],
    active_recording_mean: np.ndarray,
    reference_mean: np.ndarray | None = None,
    pca_span_percent: float = 100.0,
) -> list[Calibration]:
    """Fit one regression per selected segment with independent mean/PCA settings."""
    calibrations = []

    if MEAN_SOURCE == "reference":
        if reference_mean is None:
            raise ValueError("Set REFERENCE_RADAR_PATH when MEAN_SOURCE is 'reference'.")
        feature_means = [reference_mean] * len(segments)
    elif MEAN_SOURCE == "global":
        feature_means = [active_recording_mean] * len(segments)
    else:
        feature_means = [np.mean(segment.features, axis=0) for segment in segments]

    if PCA_MODE == "global":
        centered_calibration_data = np.vstack(
            [
                segment.features - feature_mean
                for segment, feature_mean in zip(segments, feature_means, strict=True)
            ]
        )
        global_pca = make_pca(centered_calibration_data)

    for segment, feature_mean in zip(segments, feature_means, strict=True):
        if PCA_MODE == "global":
            pca = global_pca
        else:
            pca = make_pca(segment.features - feature_mean)

        model, fit_components, fit_positions = fit_motor_model(
            segment,
            pca,
            feature_mean,
            pca_span_percent,
        )
        calibrations.append(
            Calibration(
                segment.index,
                segment.label,
                feature_mean,
                pca,
                model,
                fit_components,
                fit_positions,
                pca_span_percent,
            )
        )

    return calibrations


def refit_calibrations_for_span(
    pca_calibrations: list[Calibration],
    segments: list[Segment],
    pca_span_percent: float,
) -> list[Calibration]:
    """Refit regressions at another PCA span while reusing the fitted PCAs."""
    segments_by_index = {segment.index: segment for segment in segments}
    calibrations = []
    for pca_calibration in pca_calibrations:
        segment = segments_by_index[pca_calibration.segment_index]
        model, fit_components, fit_positions = fit_motor_model(
            segment,
            pca_calibration.pca,
            pca_calibration.feature_mean,
            pca_span_percent,
        )
        calibrations.append(
            Calibration(
                segment_index=pca_calibration.segment_index,
                segment_label=pca_calibration.segment_label,
                feature_mean=pca_calibration.feature_mean,
                pca=pca_calibration.pca,
                motor_model=model,
                fit_components=fit_components,
                fit_positions=fit_positions,
                pca_span_percent=pca_span_percent,
            )
        )
    return calibrations


def save_regression_model_plots(calibrations: list[Calibration]) -> None:
    """Save each fitted regression with its PCA points and motor targets."""
    REGRESSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_models_figure, all_models_axis = plt.subplots(figsize=(10, 7))
    all_components = np.concatenate(
        [calibration.fit_components for calibration in calibrations]
    )
    all_x_values = np.linspace(all_components.min(), all_components.max(), 200)

    for calibration in calibrations:
        model = calibration.motor_model
        slope = model.coef_[0]
        intercept = model.intercept_
        x_line = np.linspace(
            calibration.fit_components.min(),
            calibration.fit_components.max(),
            100,
        )
        y_line = model.predict(x_line.reshape(-1, 1))

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(
            calibration.fit_components,
            calibration.fit_positions,
            s=10,
            alpha=0.55,
            label="Calibration data used for fit",
        )
        axis.plot(x_line, y_line, color="tab:orange", label="Linear regression")
        axis.set_xlabel("First PCA component")
        axis.set_ylabel("Motor position (scaled)")
        axis.set_title(
            f"Segment {calibration.segment_index} ({calibration.segment_label})"
            f"\nSlope={slope:.6f}, intercept={intercept:.6f}"
            f"\nMean={MEAN_SOURCE}, PCA={PCA_MODE}, "
            f"retained PCA span={calibration.pca_span_percent:g}%"
        )
        axis.legend()
        axis.grid(True)
        figure.tight_layout()
        figure.savefig(
            REGRESSION_MODELS_DIR
            / f"regression_segment_{calibration.segment_index}.png",
            dpi=150,
        )
        plt.close(figure)

        all_models_axis.plot(
            all_x_values,
            model.predict(all_x_values.reshape(-1, 1)),
            label=(
                f"Segment {calibration.segment_index}: "
                f"slope={slope:.4f}, intercept={intercept:.4f}"
            ),
        )

    all_models_axis.set_xlabel("First PCA component")
    all_models_axis.set_ylabel("Motor position (scaled)")
    all_models_axis.set_title(
        f"Regression models (mean={MEAN_SOURCE}, PCA={PCA_MODE}, "
        f"retained PCA spans={', '.join(f'{item.pca_span_percent:g}%' for item in calibrations)})"
    )
    all_models_axis.legend()
    all_models_axis.grid(True)
    all_models_figure.tight_layout()
    all_models_figure.savefig(REGRESSION_MODELS_DIR / "all_models.png", dpi=150)
    plt.close(all_models_figure)


def save_model_performance_plot(
    evaluation: Evaluation,
    calibrations: list[Calibration],
) -> None:
    """Save each calibration model's mean MAE across all test segments."""
    MODEL_PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    mean_mae = np.mean(evaluation.mae, axis=1)
    labels = [str(calibration.segment_index) for calibration in calibrations]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, mean_mae)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_xlabel("Calibration model segment")
    axis.set_ylabel("Mean absolute error")
    axis.set_title(
        f"Mean MAE across all test segments "
        f"(mean={MEAN_SOURCE}, PCA={PCA_MODE})"
    )
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(MODEL_PERFORMANCE_DIR / "overall_mae_by_model.png", dpi=150)
    plt.close(figure)

    np.savetxt(
        MODEL_PERFORMANCE_DIR / "overall_mae_by_model.csv",
        np.column_stack(
            (
                [calibration.segment_index for calibration in calibrations],
                mean_mae,
            )
        ),
        delimiter=",",
        header="calibration_segment,mean_mae_across_test_segments",
        comments="",
    )


def save_trimming_comparison_plot(
    percentages: tuple[float, ...],
    calibrations_by_percentage: list[list[Calibration]],
    evaluations: list[Evaluation],
) -> None:
    """Plot each calibration model's mean test MAE for each retained PCA span."""
    MODEL_PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    segment_indices = [item.segment_index for item in calibrations_by_percentage[0]]
    mean_mae_by_percentage = np.array(
        [np.mean(evaluation.mae, axis=1) for evaluation in evaluations]
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for row, percentage in enumerate(percentages):
        axis.plot(
            segment_indices,
            mean_mae_by_percentage[row],
            marker="o",
            label=f"{percentage:g}% PCA span",
        )

    axis.set_xlabel("Calibration model segment")
    axis.set_ylabel("Mean MAE across all test segments")
    axis.set_title(
        f"Calibration performance by retained PCA span "
        f"(mean={MEAN_SOURCE}, PCA={PCA_MODE})"
    )
    axis.set_xticks(segment_indices)
    axis.legend(title="Retained span")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        MODEL_PERFORMANCE_DIR / "mae_by_model_and_pca_span.png",
        dpi=150,
    )
    plt.close(figure)

    csv_data = np.column_stack((segment_indices, mean_mae_by_percentage.T))
    csv_header = "calibration_segment," + ",".join(
        f"mean_mae_{percentage:g}_percent" for percentage in percentages
    )
    np.savetxt(
        MODEL_PERFORMANCE_DIR / "mae_by_model_and_pca_span.csv",
        csv_data,
        delimiter=",",
        header=csv_header,
        comments="",
    )


def select_calibration_segments(segments: list[Segment]) -> list[Segment]:
    """Return only the configured calibration segments."""
    if CALIBRATION_SEGMENT_INDICES is None:
        return segments

    if len(set(CALIBRATION_SEGMENT_INDICES)) != len(CALIBRATION_SEGMENT_INDICES):
        raise ValueError("CALIBRATION_SEGMENT_INDICES must not contain duplicates.")

    segments_by_index = {segment.index: segment for segment in segments}
    missing_indices = [
        index
        for index in CALIBRATION_SEGMENT_INDICES
        if index not in segments_by_index
    ]
    if missing_indices:
        available_indices = sorted(segments_by_index)
        raise ValueError(
            f"Unknown calibration segment indices: {missing_indices}. "
            f"Available segments: {available_indices}"
        )

    return [segments_by_index[index] for index in CALIBRATION_SEGMENT_INDICES]


def evaluate_calibrations(
    calibrations: list[Calibration],
    test_segments: list[Segment],
) -> Evaluation:
    """Calculate error matrices and predictions for every calibration/test pair."""
    rmse = np.empty((len(calibrations), len(test_segments)), dtype=float)
    mae = np.empty_like(rmse)
    max_error = np.empty_like(rmse)
    predictions = []

    for row, calibration in enumerate(calibrations):
        calibration_predictions = []
        for column, segment in enumerate(test_segments):
            component = apply_pca(
                calibration.pca,
                segment.features - calibration.feature_mean,
            )
            prediction = calibration.motor_model.predict(component.reshape(-1, 1))
            prediction_at_motor_times = np.interp(
                segment.motor_times,
                segment.radar_times,
                prediction,
            )
            actual = segment.motor_positions / MOTOR_POSITION_SCALE
            absolute_error = np.abs(actual - prediction_at_motor_times)
            rmse[row, column] = np.sqrt(np.mean(absolute_error**2))
            mae[row, column] = np.mean(absolute_error)
            max_error[row, column] = np.max(absolute_error)
            calibration_predictions.append(prediction)
        predictions.append(calibration_predictions)

    return Evaluation(rmse, mae, max_error, predictions)


def save_metric_plot(
    values: np.ndarray,
    metric_name: str,
    colorbar_label: str,
    filename: str,
    calibrations: list[Calibration],
    test_segments: list[Segment],
) -> None:
    """Save one calibration-to-test metric matrix."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(values, aspect="auto")
    x_tick_positions = list(range(0, len(test_segments), 5))
    axis.set_xticks(
        x_tick_positions,
        [test_segments[position].index for position in x_tick_positions],
    )
    axis.set_yticks(
        range(len(calibrations)),
        [calibration.segment_index for calibration in calibrations],
    )
    axis.set_xlabel("Test segment")
    axis.set_ylabel("Calibration segment")
    axis.set_title(
        f"Calibration transfer {metric_name} "
        f"(mean={MEAN_SOURCE}, PCA={PCA_MODE})"
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(figure)


def save_segment_prediction_plots(
    evaluation: Evaluation,
    calibrations: list[Calibration],
    test_segments: list[Segment],
) -> None:
    """Save actual motor and predicted movement plots for every model/segment."""
    PREDICTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for calibration_row, calibration in enumerate(calibrations):
        for test_column, segment in enumerate(test_segments):
            prediction = evaluation.predictions[calibration_row][test_column]
            prediction_at_motor_times = np.interp(
                segment.motor_times,
                segment.radar_times,
                prediction,
            )
            actual = segment.motor_positions / MOTOR_POSITION_SCALE

            figure, axis = plt.subplots(figsize=(10, 5))
            axis.plot(
                segment.motor_times,
                actual,
                label="Measured motor position",
                linewidth=2,
            )
            axis.plot(
                segment.radar_times,
                prediction,
                label="Predicted movement",
                color="tab:orange",
                alpha=0.85,
            )
            axis.plot(
                segment.motor_times,
                prediction_at_motor_times,
                label="Prediction at motor samples",
                color="tab:green",
                alpha=0.85,
            )
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Motor position (scaled)")
            axis.set_title(
                f"Calibration {calibration.segment_index} -> Test {segment.index} "
                f"| RMSE={evaluation.rmse[calibration_row, test_column]:.4f}, "
                f"MAE={evaluation.mae[calibration_row, test_column]:.4f}, "
                f"Max error={evaluation.max_error[calibration_row, test_column]:.4f}"
            )
            axis.legend()
            axis.grid(True)
            figure.tight_layout()
            figure.savefig(
                PREDICTION_PLOTS_DIR
                / f"cal_{calibration.segment_index}_test_{segment.index}.png",
                dpi=150,
            )
            plt.close(figure)


def main() -> None:
    """Run calibration and report the transfer errors."""
    if MEAN_SOURCE == "reference" and REFERENCE_RADAR_PATH is None:
        raise ValueError("Set REFERENCE_RADAR_PATH when MEAN_SOURCE is 'reference'.")
    if not CALIBRATION_PCA_SPAN_PERCENTAGES:
        raise ValueError("CALIBRATION_PCA_SPAN_PERCENTAGES must contain at least one value.")
    if len(set(CALIBRATION_PCA_SPAN_PERCENTAGES)) != len(CALIBRATION_PCA_SPAN_PERCENTAGES):
        raise ValueError("CALIBRATION_PCA_SPAN_PERCENTAGES must not contain duplicates.")
    if any(not 0.0 < value <= 100.0 for value in CALIBRATION_PCA_SPAN_PERCENTAGES):
        raise ValueError("Every retained PCA-span percentage must be in (0, 100].")

    test_recording = load_recording(RADAR_PATH, MOTOR_PATH)
    calibration_segments = select_calibration_segments(
        split_into_segments(test_recording)
    )
    test_segments = split_into_segments(test_recording)
    if not calibration_segments or not test_segments:
        raise ValueError("No complete segments found; check the segment length and logs.")

    active_recording_mean = np.mean(test_recording.features, axis=0)
    reference_mean = None
    if MEAN_SOURCE == "reference":
        reference_features = load_radar_features(REFERENCE_RADAR_PATH)
        reference_mean = np.mean(reference_features, axis=0)

    pca_calibrations = fit_calibrations(
        calibration_segments,
        active_recording_mean,
        reference_mean,
        CALIBRATION_PCA_SPAN_PERCENTAGES[0],
    )
    calibrations_by_percentage = [pca_calibrations]
    evaluations = [evaluate_calibrations(pca_calibrations, test_segments)]
    for percentage in CALIBRATION_PCA_SPAN_PERCENTAGES[1:]:
        calibrations = refit_calibrations_for_span(
            pca_calibrations,
            calibration_segments,
            percentage,
        )
        calibrations_by_percentage.append(calibrations)
        evaluations.append(evaluate_calibrations(calibrations, test_segments))

    # Save the detailed matrices and per-segment plots for the first configured
    # percentage; the comparison plot below covers every configured percentage.
    calibrations = calibrations_by_percentage[0]
    evaluation = evaluations[0]
    save_regression_model_plots(calibrations)
    save_metric_plot(
        evaluation.rmse,
        "RMSE",
        "RMSE",
        "calibration_matrix_rmse.png",
        calibrations,
        test_segments,
    )
    save_metric_plot(
        evaluation.mae,
        "MAE",
        "MAE",
        "calibration_matrix_mae.png",
        calibrations,
        test_segments,
    )
    save_metric_plot(
        evaluation.max_error,
        "maximum absolute error",
        "Maximum absolute error",
        "calibration_matrix_max_error.png",
        calibrations,
        test_segments,
    )
    save_segment_prediction_plots(evaluation, calibrations, test_segments)
    save_model_performance_plot(evaluation, calibrations)
    save_trimming_comparison_plot(
        CALIBRATION_PCA_SPAN_PERCENTAGES,
        calibrations_by_percentage,
        evaluations,
    )

    print("Calibration segment indices:", [item.segment_index for item in calibrations])
    print("Calibration source segments:", [segment.index for segment in calibration_segments])
    print("Test segments:", [(item.index, item.label) for item in test_segments])
    print("RMSE matrix:\n", evaluation.rmse)
    print("MAE matrix:\n", evaluation.mae)
    print("Maximum error matrix:\n", evaluation.max_error)
    print(f"Mean source: {MEAN_SOURCE}; PCA mode: {PCA_MODE}")
    print(f"Saved matrices to {RESULTS_DIR}")
    print(f"Saved segment predictions to {PREDICTION_PLOTS_DIR}")
    print(f"Saved regression model plots to {REGRESSION_MODELS_DIR}")
    print(f"Saved model performance to {MODEL_PERFORMANCE_DIR}")
    print(
        "Compared retained PCA spans:",
        [f"{percentage:g}%" for percentage in CALIBRATION_PCA_SPAN_PERCENTAGES],
    )


if __name__ == "__main__":
    main()
