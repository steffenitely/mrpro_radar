import json
from pathlib import Path

import numpy as np

data_path_static = Path("Data/Input/CalibrationTrial/RadarLogs/radar_20261022_143214.npz")
data_path_dynamic = Path("Data/Input/CalibrationTrial/RadarLogs/radar_20261022_150910.npz")
motor_path = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_150932.json")
output_dir = Path("Data/Output/FirstMovement/Visualizations")


def load_radar(path: Path):
    with np.load(path) as data:
        radar_cube = data["radar_cube"]
        time_cube = data["time_cube"]
    return radar_cube, time_cube


def load_motor_log(path: Path):
    with path.open(encoding="utf-8") as file:
        log_data = json.load(file)

    motor_log = log_data.get("motor_log")
    if not isinstance(motor_log, list):
        raise ValueError(f"Expected a 'motor_log' list in {path}.")

    position_samples = [
        (entry["t"], entry["position"])
        for entry in motor_log
        if isinstance(entry, dict)
        and isinstance(entry.get("t"), (int, float))
        and isinstance(entry.get("position"), (int, float))
        and np.isfinite(entry["t"])
        and np.isfinite(entry["position"])
    ]
    if len(position_samples) < 2:
        raise ValueError(f"Expected at least two numeric motor position samples in {path}.")

    position_samples.sort(key=lambda sample: sample[0])
    motor_times, motor_positions = np.asarray(position_samples, dtype=float).T
    return motor_times, motor_positions


def plot_radargram(radar_cube, start_frame, end_frame, output_path):
    import matplotlib.pyplot as plt

    selected = radar_cube[:, start_frame:end_frame, :]
    vmin = selected.min()
    vmax = selected.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for channel, ax in enumerate(axes):
        image = ax.imshow(
            selected[channel],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            extent=(-0.5, radar_cube.shape[2] - 0.5, end_frame - 0.5, start_frame - 0.5),
        )
        ax.set_title(f"Channel {channel + 1} (Correlation)")
        ax.set_xlabel("Correlation bin")
        ax.set_ylabel("Frame")

    fig.colorbar(image, ax=axes, location="right", shrink=0.9, pad=0.02, label="Correlation strength")
    fig.suptitle(f"Radargram frames {start_frame}–{end_frame - 1}")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved radargram: {output_path}")
    plt.show()


def plot_dynamic_minus_static(
    static_cube,
    dynamic_cube,
    dynamic_time_cube,
    motor_times,
    motor_positions,
    output_path,
):
    import matplotlib.pyplot as plt

    start_time = 15
    end_time = 20
    bin_start = 50
    bin_end = 150
    channel = 1

    if static_cube.ndim != 3 or dynamic_cube.ndim != 3:
        raise ValueError("Expected static and dynamic radar cubes with shape (2, frames, samples).")
    if static_cube.shape[0] != dynamic_cube.shape[0] or static_cube.shape[2] != dynamic_cube.shape[2]:
        raise ValueError(
            "Static and dynamic recordings must have the same channel and sample dimensions; "
            f"got {static_cube.shape} and {dynamic_cube.shape}."
        )
    if dynamic_time_cube.shape != dynamic_cube.shape[:2]:
        raise ValueError(
            f"Expected dynamic time_cube shape {dynamic_cube.shape[:2]}, got {dynamic_time_cube.shape}."
        )

    static_mean = static_cube.mean(axis=1)
    difference = dynamic_cube - static_mean[:, np.newaxis, :]
    time_mask = (dynamic_time_cube[channel] >= start_time) & (dynamic_time_cube[channel] <= end_time)
    if not time_mask.any():
        raise ValueError(f"No dynamic radar frames found from {start_time} to {end_time} seconds.")

    selected = difference[channel, time_mask, bin_start : bin_end + 1]
    color_limit = np.percentile(np.abs(selected), 99)
    if color_limit == 0:
        color_limit = np.abs(selected).max() or 1.0

    sample_bins = np.arange(bin_start, bin_end + 1)
    if motor_times[0] > start_time or motor_times[-1] < end_time:
        raise ValueError(
            f"Motor log ({motor_times[0]:.3f}–{motor_times[-1]:.3f} s) "
            f"does not cover the requested {start_time}–{end_time} s interval."
        )

    interpolation_times = np.linspace(start_time, end_time, 1000)
    interpolated_positions = np.interp(interpolation_times, motor_times, motor_positions)
    motor_mask = (motor_times >= start_time) & (motor_times <= end_time)

    with plt.rc_context({"font.family": "serif", "font.size": 10, "axes.labelsize": 11}):
        fig, (ax, position_ax) = plt.subplots(
            2,
            1,
            figsize=(7.2, 6.2),
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        image = ax.pcolormesh(
            dynamic_time_cube[channel, time_mask],
            sample_bins,
            selected.T,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            shading="nearest",
            rasterized=True,
        )
        ax.set_title(f"Dynamic − static mean, receiver channel {channel + 1}")
        ax.set_ylabel("Correlation bin")
        ax.set_xlim(start_time, end_time)
        ax.set_ylim(bin_start - 0.5, bin_end + 0.5)
        ax.set_xticks(np.arange(start_time, end_time + 0.1, 1))
        ax.set_yticks(np.arange(bin_start, bin_end + 1, 10))
        ax.grid(axis="x", color="0.5", linewidth=0.5, alpha=0.22)
        ax.set_axisbelow(True)
        colorbar = fig.colorbar(image, ax=ax, pad=0.025)
        colorbar.set_label("Correlation difference")
        position_ax.plot(
            interpolation_times,
            interpolated_positions,
            color="#2b6f9c",
            linewidth=1.8,
            label="Linear interpolation",
        )
        position_ax.scatter(
            motor_times[motor_mask],
            motor_positions[motor_mask],
            color="#b24a3a",
            s=12,
            zorder=3,
            label="Logged positions",
        )
        position_ax.set_ylabel("Motor position\n(controller units)")
        position_ax.set_xlabel("Time (s)")
        position_ax.set_xlim(start_time, end_time)
        position_ax.set_xticks(np.arange(start_time, end_time + 0.1, 1))
        position_ax.grid(True, color="0.5", linewidth=0.5, alpha=0.22)
        position_ax.legend(frameon=False, loc="best")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved dynamic-minus-static radargram: {output_path}")
        print(f"Saved thesis-ready vector plot: {pdf_path}")
    plt.show()


def plot_motor_position(motor_times, motor_positions, output_path):
    import matplotlib.pyplot as plt

    start_time, end_time = 15, 20
    in_range = (motor_times >= start_time) & (motor_times <= end_time)
    if motor_times[0] > start_time or motor_times[-1] < end_time:
        raise ValueError(f"Motor log does not cover {start_time}–{end_time} seconds.")

    interpolation_times = np.linspace(start_time, end_time, 1000)
    interpolated_positions = np.interp(interpolation_times, motor_times, motor_positions)
    with plt.rc_context({"font.family": "serif", "font.size": 10, "axes.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
        ax.plot(
            interpolation_times,
            interpolated_positions,
            color="#2b6f9c",
            linewidth=1.8,
            label="Linear interpolation",
        )
        ax.scatter(
            motor_times[in_range],
            motor_positions[in_range],
            color="#b24a3a",
            edgecolor="white",
            linewidth=0.4,
            s=18,
            zorder=3,
            label="Logged positions",
        )
        ax.set_title("Motor position during target movement")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motor position (controller units)")
        ax.set_xlim(start_time, end_time)
        ax.set_xticks(np.arange(start_time, end_time + 0.1, 1))
        ax.grid(True, color="0.5", linewidth=0.5, alpha=0.22)
        ax.legend(frameon=False, loc="best")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved motor-position plot: {output_path}")
        print(f"Saved vector plot: {pdf_path}")
    plt.show()


def plot_overlaid_radar_and_motor(
    static_cube,
    dynamic_cube,
    dynamic_time_cube,
    motor_times,
    motor_positions,
    output_path,
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    start_time, end_time = 15, 20
    bin_start, bin_end = 50, 150
    channel = 1
    time_mask = (dynamic_time_cube[channel] >= start_time) & (dynamic_time_cube[channel] <= end_time)
    if not time_mask.any():
        raise ValueError(f"No radar frames found from {start_time} to {end_time} seconds.")
    if motor_times[0] > start_time or motor_times[-1] < end_time:
        raise ValueError(f"Motor log does not cover {start_time}–{end_time} seconds.")

    difference = dynamic_cube[channel] - static_cube[channel].mean(axis=0)
    selected = difference[time_mask, bin_start : bin_end + 1]
    color_limit = np.percentile(np.abs(selected), 99)
    if color_limit == 0:
        color_limit = np.abs(selected).max() or 1.0
    sample_bins = np.arange(bin_start, bin_end + 1)
    interpolation_times = np.linspace(start_time, end_time, 1000)
    interpolated_positions = np.interp(interpolation_times, motor_times, motor_positions)
    motor_mask = (motor_times >= start_time) & (motor_times <= end_time)

    with plt.rc_context({"font.family": "serif", "font.size": 10, "axes.labelsize": 11}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        image = ax.pcolormesh(
            dynamic_time_cube[channel, time_mask],
            sample_bins,
            selected.T,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            shading="nearest",
            rasterized=True,
        )
        ax.set_title("Radar difference and motor position, receiver channel 2")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Correlation bin")
        ax.set_xlim(start_time, end_time)
        ax.set_ylim(bin_start - 0.5, bin_end + 0.5)
        ax.set_xticks(np.arange(start_time, end_time + 0.1, 1))
        ax.set_yticks(np.arange(bin_start, bin_end + 1, 10))
        ax.grid(axis="x", color="0.5", linewidth=0.5, alpha=0.22)
        ax.set_axisbelow(True)

        position_ax = ax.twinx()
        line, = position_ax.plot(
            interpolation_times,
            interpolated_positions,
            color="#f0c419",
            linewidth=2,
            label="Interpolated motor position",
            zorder=4,
        )
        line.set_path_effects([path_effects.Stroke(linewidth=3.5, foreground="#333333"), path_effects.Normal()])
        position_ax.scatter(
            motor_times[motor_mask],
            motor_positions[motor_mask],
            color="#f0c419",
            edgecolor="#333333",
            linewidth=0.5,
            s=18,
            label="Logged positions",
            zorder=5,
        )
        position_ax.set_ylim(0, max(1, float(np.max(motor_positions[motor_mask])) * 1.08))
        position_ax.set_ylabel("Motor position (controller units)")
        colorbar = fig.colorbar(image, ax=ax, pad=0.12)
        colorbar.set_label("Dynamic − static correlation")
        position_ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved overlaid radar and motor plot: {output_path}")
        print(f"Saved vector plot: {pdf_path}")
    plt.show()


def main():
    static_cube, _ = load_radar(data_path_static)
    dynamic_cube, dynamic_time_cube = load_radar(data_path_dynamic)
    motor_times, motor_positions = load_motor_log(motor_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{data_path_dynamic.stem}_minus_{data_path_static.stem}_channel1_bins50-150_15-20s_with_motor.png"
    )
    motor_output_path = output_dir / f"{data_path_dynamic.stem}_motor_position_15-20s.png"
    overlay_output_path = output_dir / (
        f"{data_path_dynamic.stem}_minus_{data_path_static.stem}_channel1_bins50-150_15-20s_overlay.png"
    )

    print(f"Static radar cube shape: {static_cube.shape}")
    print(f"Dynamic radar cube shape: {dynamic_cube.shape}")
    plot_dynamic_minus_static(
        static_cube,
        dynamic_cube,
        dynamic_time_cube,
        motor_times,
        motor_positions,
        output_path,
    )
    plot_motor_position(motor_times, motor_positions, motor_output_path)
    plot_overlaid_radar_and_motor(
        static_cube,
        dynamic_cube,
        dynamic_time_cube,
        motor_times,
        motor_positions,
        overlay_output_path,
    )


if __name__ == "__main__":
    main()
