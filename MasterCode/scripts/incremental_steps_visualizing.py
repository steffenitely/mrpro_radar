import argparse
from pathlib import Path

import numpy as np

data_path = Path("Data/Input/IncrementalSteps/RadarLogs/radar_20261026_125906.npz")

output_dir = Path("Data/Output/IncrementalSteps/Visualizations")


def load_radar(path: Path):
    with np.load(path) as data:
        radar_cube = data["radar_cube"]
        time_cube = data["time_cube"]
    return radar_cube, time_cube


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
        ax.set_title(f"Channel {channel} (Correlation)")
        ax.set_xlabel("Correlation bin")
        ax.set_ylabel("Frame")

    fig.colorbar(image, ax=axes, location="right", shrink=0.9, pad=0.02, label="Correlation strength")
    fig.suptitle(f"Radargram frames {start_frame}–{end_frame - 1}")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved radargram: {output_path}")
    plt.show()


def plot_radargram_over_time(radar_cube, time_cube, start_time, duration, output_path):
    import matplotlib.pyplot as plt

    bin_start = 110
    bin_end = 130
    end_time = start_time + duration
    time_masks = [(times >= start_time) & (times <= end_time) for times in time_cube]
    if any(not mask.any() for mask in time_masks):
        raise ValueError(f"No radar frames found in the time window {start_time}–{end_time} seconds.")

    selected = np.stack(
        [radar_cube[channel, time_masks[channel], bin_start : bin_end + 1] for channel in range(2)]
    )
    vmin = selected.min()
    vmax = selected.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    sample_bins = np.arange(bin_start, bin_end + 1)

    for channel, ax in enumerate(axes):
        times = time_cube[channel, time_masks[channel]]
        image = ax.pcolormesh(
            times,
            sample_bins,
            selected[channel].T,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
        ax.set_title(f"Channel {channel} (Correlation)")
        ax.set_xlabel("Time (seconds)")
        ax.set_xlim(start_time, end_time)
        ax.set_ylim(bin_start - 0.5, bin_end + 0.5)
        ax.set_xticks(np.arange(np.ceil(start_time), end_time + 1, 1))

    axes[0].set_ylabel("Correlation bin")
    fig.colorbar(image, ax=axes, location="right", shrink=0.9, pad=0.02, label="Correlation strength")
    fig.suptitle(f"Radargram from {start_time:g} to {end_time:g} seconds")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved time-based radargram: {output_path}")
    plt.show()


def plot_bin_over_time(radar_cube, time_cube, start_time, duration, correlation_bin, output_path):
    import matplotlib.pyplot as plt

    end_time = start_time + duration
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)

    for channel in range(2):
        time_mask = (time_cube[channel] >= start_time) & (time_cube[channel] <= end_time)
        if not time_mask.any():
            raise ValueError(f"No radar frames found for channel {channel} in the selected time window.")
        values = radar_cube[channel, time_mask, correlation_bin]
        axes[channel].plot(
            time_cube[channel, time_mask],
            values,
        )
        axes[channel].set_title(f"Channel {channel}")
        axes[channel].set_ylabel("Correlation strength")
        margin = max(np.ptp(values) * 0.1, abs(values.mean()) * 0.01, 1e-9)
        axes[channel].set_ylim(values.min() - margin, values.max() + margin)
        axes[channel].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xlim(start_time, end_time)
    axes[-1].set_xticks(np.arange(np.ceil(start_time), end_time + 1, 1))
    fig.suptitle(f"Correlation bin {correlation_bin} over time")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved single-bin time plot: {output_path}")
    plt.show()


def plot_window_changes(radar_cube, time_cube, window_seconds, output_path):
    import matplotlib.pyplot as plt

    start_time = time_cube.min()
    end_time = time_cube.max()
    window_edges = np.arange(start_time, end_time + window_seconds, window_seconds)
    if window_edges[-1] <= end_time:
        window_edges = np.append(window_edges, window_edges[-1] + window_seconds)

    window_means = []
    for channel in range(2):
        means = []
        for window_start, window_end in zip(window_edges[:-1], window_edges[1:]):
            mask = (time_cube[channel] >= window_start) & (time_cube[channel] < window_end)
            if not mask.any():
                means.append(np.full(radar_cube.shape[2], np.nan))
            else:
                means.append(radar_cube[channel, mask].mean(axis=0))
        window_means.append(np.stack(means))

    changes = np.stack([np.diff(channel_means, axis=0) for channel_means in window_means])
    finite_changes = np.abs(changes[np.isfinite(changes)])
    if finite_changes.size == 0:
        raise ValueError("Could not calculate window-to-window radar changes.")
    color_limit = np.percentile(finite_changes, 99)
    if color_limit == 0:
        color_limit = finite_changes.max() or 1.0

    change_times = window_edges[1:-1]
    sample_bins = np.arange(radar_cube.shape[2])
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True, sharey=True)
    for channel, ax in enumerate(axes):
        image = ax.pcolormesh(
            change_times,
            sample_bins,
            changes[channel].T,
            cmap="seismic",
            vmin=-color_limit,
            vmax=color_limit,
            shading="nearest",
        )
        ax.set_title(f"Channel {channel}")
        ax.set_ylabel("Correlation bin")
        ax.set_ylim(-0.5, radar_cube.shape[2] - 0.5)
        ax.set_yticks(np.arange(0, radar_cube.shape[2], 50))
        for movement_time in np.arange(start_time + window_seconds, end_time, window_seconds):
            ax.axvline(movement_time, color="black", linewidth=0.4, alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xlim(start_time, end_time)
    axes[-1].set_xticks(np.arange(np.ceil(start_time), end_time + 1, 2))
    fig.colorbar(image, ax=axes, location="right", shrink=0.9, pad=0.02, label="Change in correlation strength")
    fig.suptitle(f"Radar change between consecutive {window_seconds:g}-second windows")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved full-run change radargram: {output_path}")
    plt.show()


def plot_mean_subtracted_radargram(radar_cube, time_cube, output_path):
    import matplotlib.pyplot as plt

    mean_subtracted = radar_cube - radar_cube.mean(axis=1, keepdims=True)
    color_limit = np.percentile(np.abs(mean_subtracted), 99)
    if color_limit == 0:
        color_limit = np.abs(mean_subtracted).max() or 1.0

    sample_bins = np.arange(radar_cube.shape[2])
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True, sharex=True, sharey=True)
    for channel, ax in enumerate(axes):
        image = ax.pcolormesh(
            time_cube[channel],
            sample_bins,
            mean_subtracted[channel].T,
            cmap="seismic",
            vmin=-color_limit,
            vmax=color_limit,
            shading="nearest",
        )
        ax.set_title(f"Channel {channel}")
        ax.set_ylabel("Correlation bin")
        ax.set_ylim(-0.5, radar_cube.shape[2] - 0.5)
        ax.set_yticks(np.arange(0, radar_cube.shape[2], 50))

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xlim(time_cube.min(), time_cube.max())
    axes[-1].set_xticks(np.arange(0, np.ceil(time_cube.max()) + 1, 2))
    fig.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.9,
        pad=0.02,
        label="Correlation strength minus per-bin time mean",
    )
    fig.suptitle("Radargram with per-channel temporal mean subtracted")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved mean-subtracted radargram: {output_path}")
    plt.show()


def plot_mean_subtracted_zoom(radar_cube, time_cube, start_time, duration, bin_start, bin_end, output_path):
    import matplotlib.pyplot as plt

    end_time = start_time + duration
    mean_subtracted = radar_cube - radar_cube.mean(axis=1, keepdims=True)
    time_masks = [(times >= start_time) & (times <= end_time) for times in time_cube]
    if any(not mask.any() for mask in time_masks):
        raise ValueError(f"No radar frames found in the time window {start_time}–{end_time} seconds.")

    selected = np.stack(
        [mean_subtracted[channel, time_masks[channel], bin_start : bin_end + 1] for channel in range(2)]
    )
    color_limit = np.percentile(np.abs(selected), 99)
    if color_limit == 0:
        color_limit = np.abs(selected).max() or 1.0

    sample_bins = np.arange(bin_start, bin_end + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True, sharex=True, sharey=True)
    for channel, ax in enumerate(axes):
        image = ax.pcolormesh(
            time_cube[channel, time_masks[channel]],
            sample_bins,
            selected[channel].T,
            cmap="seismic",
            vmin=-color_limit,
            vmax=color_limit,
            shading="nearest",
        )
        ax.set_title(f"Channel {channel}")
        ax.set_ylabel("Correlation bin")
        ax.set_ylim(bin_start - 0.5, bin_end + 0.5)
        ax.set_yticks(np.arange(bin_start, bin_end + 1, 5))

    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_xlim(start_time, end_time)
    axes[-1].set_xticks(np.arange(np.ceil(start_time), end_time + 1, 1))
    fig.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.9,
        pad=0.02,
        label="Correlation strength minus per-bin time mean",
    )
    fig.suptitle(f"Mean-subtracted radargram: {start_time:g}–{end_time:g} seconds, bins {bin_start}–{bin_end}")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved zoomed mean-subtracted radargram: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize the two-channel radargram.")
    parser.add_argument("--data", type=Path, default=data_path, help="Path to a radar .npz file.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to plot (inclusive).")
    parser.add_argument("--end-frame", type=int, help="Stop frame (exclusive); defaults to all frames.")
    parser.add_argument("--start-time-seconds", type=float, default=0, help="Start time for the time-based view.")
    parser.add_argument("--duration-seconds", type=float, default=5, help="Duration of the time-based view.")
    parser.add_argument("--correlation-bin", type=int, default=119, help="Correlation bin to plot over time.")
    parser.add_argument("--output", type=Path, help="Output path for the frame-based radargram.")
    parser.add_argument("--time-output", type=Path, help="Output path for the time-based radargram.")
    parser.add_argument("--bin-output", type=Path, help="Output path for the single-bin time plot.")
    parser.add_argument("--change-output", type=Path, help="Output path for the full-run change radargram.")
    parser.add_argument("--mean-subtracted-output", type=Path, help="Output path for the mean-subtracted radargram.")
    parser.add_argument("--mean-zoom-start-time-seconds", type=float, default=0, help="Start time for the zoomed mean-subtracted view.")
    parser.add_argument("--mean-zoom-duration-seconds", type=float, default=5, help="Duration for the zoomed mean-subtracted view.")
    parser.add_argument("--mean-zoom-output", type=Path, help="Output path for the zoomed mean-subtracted radargram.")
    args = parser.parse_args()

    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be greater than zero.")
    if args.mean_zoom_duration_seconds <= 0:
        parser.error("--mean-zoom-duration-seconds must be greater than zero.")

    radar_cube, time_cube = load_radar(args.data)
    if radar_cube.ndim != 3 or radar_cube.shape[0] != 2:
        raise ValueError(f"Expected radar_cube shape (2, frames, samples), got {radar_cube.shape}.")
    if time_cube.shape != radar_cube.shape[:2]:
        raise ValueError(f"Expected time_cube shape {radar_cube.shape[:2]}, got {time_cube.shape}.")
    if not np.all(np.diff(time_cube, axis=1) > 0):
        raise ValueError("Expected time_cube timestamps to be strictly increasing for each channel.")
    if not 0 <= args.correlation_bin < radar_cube.shape[2]:
        parser.error(f"Correlation bin must be in [0, {radar_cube.shape[2] - 1}].")

    end_frame = radar_cube.shape[1] if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end_frame <= radar_cube.shape[1]:
        parser.error(f"Frame range must satisfy 0 <= start < end <= {radar_cube.shape[1]}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or output_dir / f"{args.data.stem}_radargram.png"
    time_output_path = args.time_output or output_dir / f"{args.data.stem}_radargram_time.png"
    bin_output_path = args.bin_output or output_dir / f"{args.data.stem}_bin_{args.correlation_bin}_over_time.png"
    change_output_path = args.change_output or output_dir / f"{args.data.stem}_window_changes.png"
    mean_subtracted_output_path = args.mean_subtracted_output or output_dir / f"{args.data.stem}_mean_subtracted.png"
    mean_zoom_output_path = args.mean_zoom_output or output_dir / f"{args.data.stem}_mean_subtracted_zoom.png"
    print(f"Radar cube shape: {radar_cube.shape}")
    print(f"Time cube shape: {time_cube.shape}")
    plot_radargram(radar_cube, args.start_frame, end_frame, output_path)
    plot_radargram_over_time(
        radar_cube,
        time_cube,
        args.start_time_seconds,
        args.duration_seconds,
        time_output_path,
    )
    plot_bin_over_time(
        radar_cube,
        time_cube,
        args.start_time_seconds,
        args.duration_seconds,
        args.correlation_bin,
        bin_output_path,
    )
    plot_window_changes(radar_cube, time_cube, 0.5, change_output_path)
    plot_mean_subtracted_radargram(radar_cube, time_cube, mean_subtracted_output_path)
    plot_mean_subtracted_zoom(
        radar_cube,
        time_cube,
        args.mean_zoom_start_time_seconds,
        args.mean_zoom_duration_seconds,
        110,
        130,
        mean_zoom_output_path,
    )


if __name__ == "__main__":
    main()