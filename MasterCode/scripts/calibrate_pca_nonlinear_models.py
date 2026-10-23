"""Compare polynomial radar-PCA-to-motor calibration models.

Edit the settings below and run this file. Each selected calibration segment
gets a separate model for every configured polynomial degree. A model is
evaluated on every segment in the active recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import calibrate_pca_centering_options as shared


# Recording paths and segmentation follow the linear calibration script.
RADAR_PATH = Path("Data/Input/CalibrationTrial/RadarLogs/radar_20261022_131201.npz")
MOTOR_PATH = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_131300.json")
REFERENCE_RADAR_PATH: Path | None = Path(
    "Data/Input/CalibrationTrial/RadarLogs/radar_20261022_143214.npz"
)

MEAN_SOURCE: Literal["global", "segmentwise", "reference"] = "reference"
PCA_MODE: Literal["global", "per_calibration_segment"] = "global"
SEGMENTATION_MODE: Literal["fixed_duration", "phase_notes"] = "phase_notes"
SEGMENT_LENGTH_SECONDS = 40.0
N_PCA_COMPONENTS = 510
MOTOR_POSITION_SCALE = 2000.0

# Degree 1 is linear; larger degrees allow progressively more curvature.
# Higher degrees can fit calibration data more closely but may overfit.
POLYNOMIAL_DEGREES: tuple[int, ...] = (1, 3)

# Choose which segments get their own calibration models. Use None for every
# segment, or list the segment indices to use (for example, (0, 1, 2, 3, 4, 5)).
CALIBRATION_SEGMENT_INDICES: tuple[int, ...] | None = (0, 1, 2, 3, 4, 5)

CHOSEN_NAME = "mid_cal"
OUTPUT_DIR = Path("Data/Output/LinearityTrial") / CHOSEN_NAME
RESULTS_DIR = OUTPUT_DIR / "calibration_matrix"
PREDICTION_PLOTS_DIR = OUTPUT_DIR / "segment_predictions"
REGRESSION_MODELS_DIR = OUTPUT_DIR / "regression_models"
MODEL_PERFORMANCE_DIR = OUTPUT_DIR / "model_performance"


@dataclass
class NonlinearCalibration:
    """A polynomial model fitted to one calibration segment."""

    segment_index: int
    segment_label: str
    degree: int
    feature_mean: np.ndarray
    pca: shared.PCACompressionOp
    motor_model: Pipeline
    fit_components: np.ndarray
    fit_positions: np.ndarray


@dataclass
class Evaluation:
    """Transfer errors and radar-time predictions for one model."""

    rmse: np.ndarray
    mae: np.ndarray
    max_error: np.ndarray
    predictions: list[np.ndarray]


def configure_shared_helpers() -> None:
    """Set shared loading and PCA helpers to this script's user options."""
    shared.RADAR_PATH = RADAR_PATH
    shared.MOTOR_PATH = MOTOR_PATH
    shared.REFERENCE_RADAR_PATH = REFERENCE_RADAR_PATH
    shared.MEAN_SOURCE = MEAN_SOURCE
    shared.PCA_MODE = PCA_MODE
    shared.SEGMENTATION_MODE = SEGMENTATION_MODE
    shared.SEGMENT_LENGTH_SECONDS = SEGMENT_LENGTH_SECONDS
    shared.N_PCA_COMPONENTS = N_PCA_COMPONENTS
    shared.MOTOR_POSITION_SCALE = MOTOR_POSITION_SCALE


def select_calibration_segments(
    segments: list[shared.Segment],
) -> list[shared.Segment]:
    """Return all segments or the configured subset, preserving chosen order."""
    if CALIBRATION_SEGMENT_INDICES is None:
        return segments

    if len(set(CALIBRATION_SEGMENT_INDICES)) != len(CALIBRATION_SEGMENT_INDICES):
        raise ValueError("CALIBRATION_SEGMENT_INDICES must not contain duplicates.")

    segments_by_index = {segment.index: segment for segment in segments}
    missing = [
        index
        for index in CALIBRATION_SEGMENT_INDICES
        if index not in segments_by_index
    ]
    if missing:
        raise ValueError(
            f"Unknown calibration segment indices: {missing}. "
            f"Available segments: {sorted(segments_by_index)}"
        )
    return [segments_by_index[index] for index in CALIBRATION_SEGMENT_INDICES]


def get_feature_means(
    calibration_segments: list[shared.Segment],
    active_recording_mean: np.ndarray,
    reference_mean: np.ndarray | None,
) -> list[np.ndarray]:
    """Choose the centering mean for each calibration segment."""
    if MEAN_SOURCE == "global":
        return [active_recording_mean] * len(calibration_segments)
    if MEAN_SOURCE == "segmentwise":
        return [
            np.mean(segment.features, axis=0) for segment in calibration_segments
        ]
    if reference_mean is None:
        raise ValueError("Set REFERENCE_RADAR_PATH when MEAN_SOURCE is 'reference'.")
    return [reference_mean] * len(calibration_segments)


def fit_nonlinear_calibrations(
    calibration_segments: list[shared.Segment],
    feature_means: list[np.ndarray],
) -> list[NonlinearCalibration]:
    """Fit every configured degree for every selected calibration segment."""
    if not POLYNOMIAL_DEGREES:
        raise ValueError("POLYNOMIAL_DEGREES must contain at least one degree.")
    if len(set(POLYNOMIAL_DEGREES)) != len(POLYNOMIAL_DEGREES):
        raise ValueError("POLYNOMIAL_DEGREES must not contain duplicates.")
    if any(
        not isinstance(degree, int) or isinstance(degree, bool) or degree < 1
        for degree in POLYNOMIAL_DEGREES
    ):
        raise ValueError("Every polynomial degree must be a positive integer.")
    if len(calibration_segments) != len(feature_means):
        raise ValueError("Each calibration segment must have exactly one feature mean.")

    if PCA_MODE == "global":
        centered_data = np.vstack(
            [
                segment.features - feature_mean
                for segment, feature_mean in zip(
                    calibration_segments, feature_means, strict=True
                )
            ]
        )
        global_pca = shared.make_pca(centered_data)

    calibrations = []
    for segment, feature_mean in zip(
        calibration_segments, feature_means, strict=True
    ):
        if PCA_MODE == "global":
            pca = global_pca
        else:
            pca = shared.make_pca(segment.features - feature_mean)

        component = shared.apply_pca(pca, segment.features - feature_mean)
        motor_on_radar = np.interp(
            segment.radar_times,
            segment.motor_times,
            segment.motor_positions,
        )
        target = motor_on_radar / MOTOR_POSITION_SCALE

        for degree in POLYNOMIAL_DEGREES:
            model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            )
            model.fit(component.reshape(-1, 1), target)
            calibrations.append(
                NonlinearCalibration(
                    segment_index=segment.index,
                    segment_label=segment.label,
                    degree=degree,
                    feature_mean=feature_mean,
                    pca=pca,
                    motor_model=model,
                    fit_components=component,
                    fit_positions=target,
                )
            )
    return calibrations


def evaluate_calibration(
    calibration: NonlinearCalibration,
    test_segments: list[shared.Segment],
) -> Evaluation:
    """Evaluate one model against the motor samples in every test segment."""
    rmse = np.empty(len(test_segments), dtype=float)
    mae = np.empty_like(rmse)
    max_error = np.empty_like(rmse)
    predictions = []

    for index, segment in enumerate(test_segments):
        component = shared.apply_pca(
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
        rmse[index] = np.sqrt(np.mean(absolute_error**2))
        mae[index] = np.mean(absolute_error)
        max_error[index] = np.max(absolute_error)
        predictions.append(prediction)

    return Evaluation(rmse, mae, max_error, predictions)


def save_metric_matrix(
    values: np.ndarray,
    metric_name: str,
    degree: int,
    calibrations: list[NonlinearCalibration],
    test_segments: list[shared.Segment],
) -> None:
    """Save the calibration-to-test error matrix for one polynomial degree."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(values, aspect="auto")
    tick_positions = list(range(0, len(test_segments), 5))
    axis.set_xticks(
        tick_positions,
        [test_segments[position].index for position in tick_positions],
    )
    calibration_indices = [item.segment_index for item in calibrations]
    axis.set_yticks(range(len(calibrations)), calibration_indices)
    axis.set_xlabel("Test segment")
    axis.set_ylabel("Calibration segment")
    axis.set_title(f"{metric_name} by calibration and test segment (degree {degree})")
    figure.colorbar(image, ax=axis, label=metric_name)
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / f"{metric_name.lower()}_degree_{degree}.png",
        dpi=150,
    )
    plt.close(figure)


def save_regression_plots(calibrations: list[NonlinearCalibration]) -> None:
    """Save each fitted curve with the calibration samples used to fit it."""
    REGRESSION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for calibration in calibrations:
        x_values = np.linspace(
            calibration.fit_components.min(),
            calibration.fit_components.max(),
            250,
        )
        y_values = calibration.motor_model.predict(x_values.reshape(-1, 1))

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(
            calibration.fit_components,
            calibration.fit_positions,
            s=10,
            alpha=0.5,
            label="Calibration data",
        )
        axis.plot(x_values, y_values, color="tab:orange", label="Polynomial model")
        axis.set_xlabel("First PCA component")
        axis.set_ylabel("Motor position (scaled)")
        axis.set_title(
            f"Calibration segment {calibration.segment_index} "
            f"({calibration.segment_label}) - polynomial degree "
            f"{calibration.degree}\nMean={MEAN_SOURCE}, PCA={PCA_MODE}"
        )
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            REGRESSION_MODELS_DIR
            / f"regression_segment_{calibration.segment_index}_degree_{calibration.degree}.png",
            dpi=150,
        )
        plt.close(figure)


def save_prediction_plots(
    calibrations: list[NonlinearCalibration],
    evaluations: list[Evaluation],
    test_segments: list[shared.Segment],
) -> None:
    """Save measured and predicted motion for each model/test-segment pair."""
    PREDICTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for calibration, evaluation in zip(calibrations, evaluations, strict=True):
        for index, segment in enumerate(test_segments):
            prediction = evaluation.predictions[index]
            prediction_at_motor_times = np.interp(
                segment.motor_times,
                segment.radar_times,
                prediction,
            )
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.plot(
                segment.motor_times,
                segment.motor_positions / MOTOR_POSITION_SCALE,
                label="Measured motor position",
                linewidth=2,
            )
            axis.plot(
                segment.radar_times,
                prediction,
                label="Predicted movement",
                color="tab:orange",
            )
            axis.plot(
                segment.motor_times,
                prediction_at_motor_times,
                label="Prediction at motor samples",
                color="tab:green",
                alpha=0.8,
            )
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Motor position (scaled)")
            axis.set_title(
                f"Calibration {calibration.segment_index} -> Test {segment.index} "
                f"(degree {calibration.degree}) | "
                f"MAE={evaluation.mae[index]:.4f}"
            )
            axis.legend()
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(
                PREDICTION_PLOTS_DIR
                / f"cal_{calibration.segment_index}_degree_{calibration.degree}"
                f"_test_{segment.index}.png",
                dpi=150,
            )
            plt.close(figure)


def save_degree_comparison(
    calibrations: list[NonlinearCalibration],
    evaluations: list[Evaluation],
) -> None:
    """Plot and save mean test MAE by model degree and calibration segment."""
    MODEL_PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    segment_indices = list(dict.fromkeys(item.segment_index for item in calibrations))
    degrees = list(POLYNOMIAL_DEGREES)
    mean_mae = np.empty((len(degrees), len(segment_indices)), dtype=float)

    for degree_index, degree in enumerate(degrees):
        for segment_index_position, segment_index in enumerate(segment_indices):
            matching = [
                evaluation.mae
                for calibration, evaluation in zip(
                    calibrations, evaluations, strict=True
                )
                if calibration.degree == degree
                and calibration.segment_index == segment_index
            ]
            mean_mae[degree_index, segment_index_position] = np.mean(
                [np.mean(segment_mae) for segment_mae in matching]
            )

    figure, axis = plt.subplots(figsize=(10, 6))
    for index, segment_index in enumerate(segment_indices):
        axis.plot(
            degrees,
            mean_mae[:, index],
            marker="o",
            label=f"Calibration segment {segment_index}",
        )
    axis.set_xticks(degrees)
    axis.set_xlabel("Polynomial degree (higher allows more curvature)")
    axis.set_ylabel("Mean absolute error across test segments")
    axis.set_title(f"Nonlinear model comparison (mean={MEAN_SOURCE}, PCA={PCA_MODE})")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        MODEL_PERFORMANCE_DIR / "mean_mae_by_polynomial_degree.png",
        dpi=150,
    )
    plt.close(figure)

    csv_data = np.array(
        [
            (degree, segment_index, mean_mae[degree_index, segment_position])
            for degree_index, degree in enumerate(degrees)
            for segment_position, segment_index in enumerate(segment_indices)
        ],
        dtype=float,
    )
    np.savetxt(
        MODEL_PERFORMANCE_DIR / "mean_mae_by_polynomial_degree.csv",
        csv_data,
        delimiter=",",
        header="polynomial_degree,calibration_segment,mean_mae_across_test_segments",
        comments="",
    )


def main() -> None:
    """Load the recording, fit each degree, evaluate, and save comparisons."""
    configure_shared_helpers()
    if MEAN_SOURCE == "reference" and REFERENCE_RADAR_PATH is None:
        raise ValueError("Set REFERENCE_RADAR_PATH when MEAN_SOURCE is 'reference'.")

    recording = shared.load_recording(RADAR_PATH, MOTOR_PATH)
    test_segments = shared.split_into_segments(recording)
    calibration_segments = select_calibration_segments(test_segments)
    if not calibration_segments or not test_segments:
        raise ValueError("No complete segments found; check the segment length and logs.")

    reference_mean = None
    if MEAN_SOURCE == "reference":
        reference_features = shared.load_radar_features(REFERENCE_RADAR_PATH)
        reference_mean = np.mean(reference_features, axis=0)
    feature_means = get_feature_means(
        calibration_segments,
        np.mean(recording.features, axis=0),
        reference_mean,
    )

    calibrations = fit_nonlinear_calibrations(calibration_segments, feature_means)
    evaluations = [
        evaluate_calibration(calibration, test_segments)
        for calibration in calibrations
    ]
    save_regression_plots(calibrations)
    save_prediction_plots(calibrations, evaluations, test_segments)

    for degree in POLYNOMIAL_DEGREES:
        degree_calibrations = [
            item for item in calibrations if item.degree == degree
        ]
        degree_evaluations = [
            evaluation
            for item, evaluation in zip(calibrations, evaluations, strict=True)
            if item.degree == degree
        ]
        save_metric_matrix(
            np.array([evaluation.rmse for evaluation in degree_evaluations]),
            "RMSE",
            degree,
            degree_calibrations,
            test_segments,
        )
        save_metric_matrix(
            np.array([evaluation.mae for evaluation in degree_evaluations]),
            "MAE",
            degree,
            degree_calibrations,
            test_segments,
        )
        save_metric_matrix(
            np.array([evaluation.max_error for evaluation in degree_evaluations]),
            "maximum_absolute_error",
            degree,
            degree_calibrations,
            test_segments,
        )

    save_degree_comparison(calibrations, evaluations)
    print("Calibration segments:", [item.index for item in calibration_segments])
    print("Polynomial degrees:", list(POLYNOMIAL_DEGREES))
    print(f"Saved calibration matrices to {RESULTS_DIR}")
    print(f"Saved movement predictions to {PREDICTION_PLOTS_DIR}")
    print(f"Saved regression curves to {REGRESSION_MODELS_DIR}")
    print(f"Saved degree comparison to {MODEL_PERFORMANCE_DIR}")


if __name__ == "__main__":
    main()
