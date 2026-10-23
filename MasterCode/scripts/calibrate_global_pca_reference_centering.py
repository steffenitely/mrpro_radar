import json
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.linear_model import LinearRegression
import joblib
from mrpro.operators.PCACompressionOp import PCACompressionOp
import torch
import os

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE — File paths
# ══════════════════════════════════════════════════════════════════════════════
# Calibrate and rnd Movement
#Both done in one measurement
RADAR = "Data/RadarTest/radar_20260729_145247.npz"
MOTOR = "Data/TimeLogs/time_log_20260729_145342.json"  


save_dir = "Data/calibration_plots/PCA_segmentation_centered/00"

motor_pca_dir = os.path.join(save_dir, "Motor_vs_PCA")
regression_dir = os.path.join(save_dir, "RegressionModels")
single_reg_dir = os.path.join(save_dir, "RegressionModel_per_segment")
cal_test_dir = os.path.join(save_dir, "Calibration_vs_Test")
cal_matrix_dir = os.path.join(save_dir, "Calibration_Matrix")

# create all folders
for d in [motor_pca_dir, regression_dir, single_reg_dir, cal_test_dir, cal_matrix_dir]:
    os.makedirs(d, exist_ok=True)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_EPOCH: datetime.datetime | None = None
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Load & parse the motor log
# ─────────────────────────────────────────────────────────────────────────────
def load_calibration_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, positions) arrays from the JSON motor log."""
    with open(path) as f:
        log = json.load(f)

    entries = log["motor_log"]
    times = np.array([e["t"] for e in entries], dtype=float)
    try:
        positions = np.array([ACTION_MAP[e["action"]] for e in entries], dtype=float)
    except KeyError as exc:
        raise KeyError(
            f"Action label {exc} is not in ACTION_MAP. "
            f"Add it with a numeric value."
        ) from exc

    return times, positions

def load_motor_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, positions) arrays from the JSON motor log."""
    with open(path) as f:
        log = json.load(f)

    entries = log["motor_log"]
    times = np.array([e["t"] for e in entries], dtype=float)
    positions = np.array([e["position"] for e in entries], dtype=float)

    return times, positions

# ─────────────────────────────────────────────────────────────────────────────
# Load Radar data
# ─────────────────────────────────────────────────────────────────────────────
def load_radar(path: str):
    data = np.load(path, allow_pickle=True) #.npy for before 9.6
    radar_cube = data["radar_cube"] #old data before 9.6. was without cube so these steps need to be excluded
    timestamps = data["timestamps"]
    time_cube = data["time_cube"]
    return radar_cube, timestamps, time_cube


# ─────────────────────────────────────────────────────────────────────────────
# Convert radar datetimes → seconds relative to radar t=0
# ─────────────────────────────────────────────────────────────────────────────
def radar_times_to_sec(radar_datetimes: list[datetime.datetime]) -> np.ndarray:
    """Subtract the first datetime so radar starts at 0."""
    t0 = radar_datetimes[0]
    return np.array([(dt - t0).total_seconds() for dt in radar_datetimes])

def allign_and_split(radar_path: str, motor_path: str, segment_length: float):
    """
    Split one radar measurement + motor log into equal time segments.

    Parameters
    ----------
    radar_path : str
        Path to radar npz file

    motor_path : str
        Path to motor json log

    segment_length : float
        Length of each segment in seconds

    Returns
    -------
    segments : list
        Each entry contains:
        (
            radar_cube_segment,
            radar_times_segment,
            motor_times_segment,
            motor_positions_segment
        )
    """

    radar_cube, timestamps, time_cube = load_radar(radar_path)
    motor_times, motor_positions = load_motor_log(motor_path)

    print("Full radar shape:", radar_cube.shape)

    # -------------------------
    # Radar time
    # -------------------------
    radar_datetimes = time_cube[0, :]
    radar_t_sec = radar_times_to_sec(radar_datetimes)


    # -------------------------
    # Align motor to radar timebase
    # -------------------------
    motor_t_aligned = align_motor_to_radar(
        motor_times,
        radar_datetimes,
        MOTOR_EPOCH
    )


    # -------------------------
    # Determine number of segments
    # -------------------------
    total_time = radar_t_sec[-1]

    n_segments = int(np.floor(total_time / segment_length))

    print(f"Total measurement time: {total_time:.2f}s")
    print(f"Creating {n_segments} segments of {segment_length}s")


    segments = []


    # -------------------------
    # Split into chunks
    # -------------------------
    for i in range(n_segments):

        start = i * segment_length
        end = (i + 1) * segment_length


        # Radar mask
        radar_mask = (
            (radar_t_sec >= start) &
            (radar_t_sec < end)
        )


        # Motor mask
        motor_mask = (
            (motor_t_aligned >= start) &
            (motor_t_aligned < end)
        )


        # Extract data
        radar_segment = radar_cube[:, radar_mask, :]
        radar_times_segment = radar_t_sec[radar_mask]

        motor_times_segment = motor_t_aligned[motor_mask]
        motor_positions_segment = motor_positions[motor_mask]


        segments.append(
            (
                radar_segment,
                radar_times_segment,
                motor_times_segment,
                motor_positions_segment
            )
        )


        print(
            f"Segment {i}: "
            f"Radar frames={radar_segment.shape[1]}, "
            f"Motor samples={len(motor_positions_segment)}, "
            f"time={start:.1f}-{end:.1f}s"
        )


    return segments

# ─────────────────────────────────────────────────────────────────────────────
# Shift motor times so they share the same t=0 as the radar
# ─────────────────────────────────────────────────────────────────────────────
def align_motor_to_radar(
    motor_times: np.ndarray,
    radar_datetimes: list[datetime.datetime],
    motor_epoch: datetime.datetime | None,
) -> np.ndarray:
    """
    Return motor times expressed in the same second-offset as radar_times_sec.

    If motor_epoch is None → assume both start at t=0 simultaneously.
    If motor_epoch is a datetime → compute offset from radar[0] to motor_epoch.
    """
    radar_t0 = radar_datetimes[0]

    if motor_epoch is None:
        offset = 0.0
    else:
        offset = (motor_epoch - radar_t0).total_seconds()

    return motor_times + offset


# ─────────────────────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────────────────────
def remove_clutter(radar_data_data):
    # radar_data: (T, R)
    return radar_data_data - np.mean(radar_data_data, axis=0, keepdims=True)

def highpass_clutter(radar_data):
    return detrend(radar_data, axis=0)

def select_range_bin(radar_data):
    energy = np.var(radar_data, axis=0)
    return np.argmax(energy)

def extract_signal(radar_data, range_bin):
    return radar_data[:, range_bin]   # shape: (T,)

def find_peaks_per_frame(radar):
    # radar: (T, R)
    peaks = np.argmax(radar, axis=1)   # shape: (T,)
    return peaks

def align_radar(radar, peaks, ref_bin=200):
    T, R = radar.shape
    aligned = np.zeros_like(radar)

    for t in range(T):
        shift = ref_bin - peaks[t]
        aligned[t] = np.roll(radar[t], shift)

    return aligned

def plot_both(data1, data2):
    rx1, rx2 = data1, data2

    # Shared color scale (important for correlation comparability)
    vmin = min(rx1.min(), rx2.min())
    vmax = max(rx1.max(), rx2.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(rx1, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[0].set_title("Channel 0 (Correlation)")

    im1 = axes[1].imshow(rx2, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[1].set_title("Channel 1 (Correlation)")

    # One shared colorbar placed on the RIGHT side
    cbar = fig.colorbar(im1, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("Correlation strength")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Basic Info Extraction
# ─────────────────────────────────────────────────────────────────────────────
def basic_info_extraction(radar_cube):
    rx1 = radar_cube[0, :, :] #old was without time stamps so directly data
    rx2 = radar_cube[1, :, :]

    rx1_cf = remove_clutter(rx1)
    rx2_cf = remove_clutter(rx2)
    rx1_cf = detrend(rx1_cf)
    rx2_cf = detrend(rx2_cf)

    #---- cut if neccassary
    #rx1_cf = rx1_cf[100:, :]
    #rx2_cf = rx2_cf[100:, :]

    peaks_rx1 = find_peaks_per_frame(rx1_cf)
    peaks_rx2 = find_peaks_per_frame(rx2_cf)

    unique_peaks1 = np.unique(peaks_rx1)
    unique_peaks2 = np.unique(peaks_rx2)

    energy1 = np.var(rx1_cf, axis=0)  # (510,)
    energy2 = np.var(rx2_cf, axis=0)

    rbin1 = select_range_bin(rx1_cf)
    rbin2 = select_range_bin(rx2_cf)
    return rx1, rx2, rx1_cf, rx2_cf, peaks_rx1, peaks_rx2, energy1, energy2, rbin1, rbin2

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def mae(y_true, y_pred):
    """Calculate mean absolute error between y_true and y_pred."""
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def max_abs_error(y_true, y_pred):
    """Calculate maximum absolute error between y_true and y_pred."""
    max_error = np.max(np.abs(y_true - y_pred))
    return max_error


# ---------------------------------  Plot of signal processing ----------------------------
# ─────────────────────────────────────────────────────────────────────────────
# Plotting of radargrams, radargrams with static removed and signal of highest energy
# ─────────────────────────────────────────────────────────────────────────────
def plot_basics(rx1, rx2, rx1_cf, rx2_cf, energy1, energy2, rbin1, rbin2):
        
    fig_basics, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)

    # shared color scale for heatmaps
    vmin = min(rx1.min(), rx2.min())
    vmax = max(rx1.max(), rx2.max())
    print(f"vmin: {vmin}, vmmax: {vmax}")

    vmin_cut = min(rx1_cf.min(), rx2_cf.min())
    vmax_cut = max(rx1_cf.max(), rx2_cf.max())

    # -------------------------
    # Row 1: raw data
    # -------------------------
    im0 = axes[0, 0].imshow(rx1, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 0].set_title("rx1 raw")

    axes[0, 1].imshow(rx2, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 1].set_title("rx2 raw")

    # -------------------------
    # Row 2: clutter removed
    # -------------------------
    axes[1, 0].imshow(rx1_cf, cmap="viridis", vmin=vmin_cut, vmax=vmax_cut, aspect="auto")
    axes[1, 0].set_title("rx1 clutter-free")

    axes[1, 1].imshow(rx2_cf, cmap="viridis", vmin=vmin_cut, vmax=vmax_cut, aspect="auto")
    axes[1, 1].set_title("rx2 clutter-free")

    # -------------------------
    # Row 3: line plots
    # -------------------------
    axes[2, 0].plot(energy1)
    axes[2, 0].set_title("rx1 energy of bins")

    axes[2, 1].plot(energy2)
    axes[2, 1].set_title("rx2 energy of bins")

    # -------------------------
    # Row 4: rbin view
    # -------------------------
    axes[3, 0].plot(rx1_cf[:, rbin1])
    axes[3, 0].set_title(f"rx1 bin:{rbin1}")

    axes[3, 1].plot(rx2_cf[:, rbin2])
    axes[3, 1].set_title(f"rx1 bin:{rbin2}")

    # colorbar for all heatmaps (rows 1–2)
    cbar = fig_basics.colorbar(im0, ax=axes[0, :], location="right", shrink=0.9)
    #cbar.set_label("Correlation strength")
    cbar = fig_basics.colorbar(im0, ax=axes[1, :], location="right", shrink=0.9)

    plt.show()

def plot_calibration_models(calibration_models):
    plt.figure(figsize=(8, 6))

    # global x-range for fair comparison
    global_min = -1   # or compute from your data once
    global_max = 1

    x_line = np.linspace(global_min, global_max, 100)
    for r in calibration_models:
        i = r["segment"]
        slope = r["slope"]
        intercept = r["intercept"]

        y_line = slope * x_line + intercept

        plt.plot(
            x_line,
            y_line,
            label=f"Seg {i} (m={slope:.3f}, b={intercept:.3f})"
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("Displacement")
    plt.title("Regression Lines (Segments 0–5)")
    plt.legend()
    plt.grid()
    
    filename = os.path.join(regression_dir,"RegressionModels.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


    segments_idx = [r["segment"] for r in calibration_models]
    slopes = [r["slope"] for r in calibration_models]

    plt.figure()
    plt.scatter(segments_idx, slopes, marker='o')
    plt.title("Calibration slope per segment (0–5)")
    plt.xlabel("Segment")
    plt.ylabel("Slope")
    plt.grid()

    filename = os.path.join(regression_dir,"Models_Slopes.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plotting Motor Movement
# ─────────────────────────────────────────────────────────────────────────────
def plot_motor(times, positions):
    # ---- plot
    plt.figure()
    plt.plot(times, positions)  # linear interpolation between points (default)
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Motor Position vs Time")
    plt.grid()
    plt.show()

def plot_motor_vs_pca(
    segment,
    radar_times_sec: np.ndarray,
    pca_values: np.ndarray,          # shape (Timeframes,) — first PC or any scalar
    motor_times_aligned: np.ndarray,
    motor_positions: np.ndarray,
    pca_label: str = "PC 1",
    position_unit: str = "position",
) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f8f8f8")
    ax1.set_facecolor("#f8f8f8")

    # ── PCA (left axis) ──────────────────────────────────────────────────────
    color_pca = "#2176AE"
    ax1.set_xlabel("Time  (s from t=0)", fontsize=12)
    ax1.set_ylabel(pca_label, color=color_pca, fontsize=11)
    ax1.plot(radar_times_sec, pca_values,
             color=color_pca, lw=1.8, label=pca_label, zorder=3)
    ax1.tick_params(axis="y", labelcolor=color_pca)
    ax1.grid(True, ls="--", alpha=0.4)

    # ── Motor position (right axis, step plot) ────────────────────────────────
    color_motor = "#E84855"
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Motor {position_unit}", color=color_motor, fontsize=11)

    #
    # build dense interpolated motor signal
    t_dense = np.linspace(motor_times_aligned[0], motor_times_aligned[-1], 1000)
    y_dense = np.interp(t_dense, motor_times_aligned, motor_positions)
    ax2.plot(t_dense, y_dense, color=color_motor, lw=2.2, label="Motor position", zorder=4, alpha=0.5)

    # mark each logged sample
    ax2.scatter(motor_times_aligned, motor_positions,
                color=color_motor, s=25, zorder=5)

    ax2.tick_params(axis="y", labelcolor=color_motor)
    ax2.yaxis.set_major_locator(ticker.AutoLocator())

    # ── Legend ────────────────────────────────────────────────────────────────
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.85)

    plt.title("PCA  vs  Motor Position  (aligned to t = 0)", fontsize=13, pad=10)
    plt.tight_layout()
    filename = os.path.join(motor_pca_dir, f"Motor_vs_PCA_segment{segment}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    


#==============================================================MAIN===========================================
#────────────────────────────────────────────────────────────────────────────


radar_cube, timestamps, time_cube = load_radar(RADAR)
motor_times, motor_positions = load_motor_log(MOTOR)

radar_datetimes = time_cube[0, :]
radar_times = radar_times_to_sec(radar_datetimes)

motor_times_aligned = align_motor_to_radar(
    motor_times,
    radar_datetimes,
    MOTOR_EPOCH
)

segment_length = 40.0

first_segment_mask = (radar_times >= 0.0) & (radar_times < segment_length)
reference_mean = None

if np.sum(first_segment_mask) >= 2:
    rx1_ref, rx2_ref, rx1_cf_ref, rx2_cf_ref, *_ = basic_info_extraction(
        radar_cube[:, first_segment_mask, :]
    )
    reference_mean = np.mean(np.hstack([rx1_cf_ref, rx2_cf_ref]), axis=0)
    print("Stored reference mean from first segment; using it for full-measurement PCA")
else:
    print("First segment has insufficient data; falling back to the overall mean")

rx1, rx2, rx1_cf, rx2_cf, *_ = basic_info_extraction(radar_cube)

M_both = np.hstack([rx1_cf, rx2_cf])
mean = reference_mean if reference_mean is not None else np.mean(M_both, axis=0)

M_centered = torch.tensor(M_both - mean, dtype=torch.float32)

op = PCACompressionOp(
    data=M_centered,
    n_components=510,
    centering=False
)

pca = op(M_centered)[0]
pc1_full = pca[:, 0].numpy()

y_aligned_full = np.interp(
    radar_times,
    motor_times_aligned,
    motor_positions
)

y_full = y_aligned_full / 2000

#
#----------------------------segmentation
total_time = radar_times[-1]
n_segments = int(np.floor(total_time / segment_length))
print(f"number of segments: {n_segments} (segment length: {segment_length}s)")

segments = []

for i in range(1, n_segments):
    
    start = i * segment_length
    end = (i + 1) * segment_length

    radar_mask = (radar_times >= start) & (radar_times < end)
    if np.sum(radar_mask) < 2:
        continue

    # -----------------------------
    # Radar/PCA data
    # -----------------------------
    radar_times_seg = radar_times[radar_mask]
    pc1_seg = pc1_full[radar_mask]
    # -----------------------------
    # Motor segmentation 
    # -----------------------------
    t_start = radar_times_seg[0]
    t_end   = radar_times_seg[-1]

    motor_mask = (motor_times_aligned >= t_start) & (motor_times_aligned <= t_end)

    motor_times_seg = motor_times_aligned[motor_mask]
    motor_positions_seg = motor_positions[motor_mask]


    # -----------------------------
    # Store everything
    # -----------------------------
    segments.append({
        "segment": i,
        "pc1": pc1_seg,
        "y": y_full[radar_mask],
        "radar_times": radar_times_seg,
        "motor_times": motor_times_seg,
        "motor_positions": motor_positions_seg
    })

    plot_motor_vs_pca(radar_times_sec=radar_times_seg,
                      pca_values=pc1_seg,
                      motor_times_aligned=motor_times_seg,
                      motor_positions=motor_positions_seg,
                      segment=i
                      )



calibration_models = []


for seg in segments[:6]:

    X = seg["pc1"]
    y = seg["y"]

    if len(X) < 10:
        continue

    # Trim extremes (same as before)
    xmin, xmax = X.min(), X.max()
    span = xmax - xmin

    lower = xmin + 0.1 * span
    upper = xmax - 0.1 * span

    mask = (X >= lower) & (X <= upper)

    X_trimmed = X[mask].reshape(-1, 1)
    y_trimmed = y[mask]


    # ─────────────────────────────────────
    # Downsample static regions (FIXED)
    # ─────────────────────────────────────
    dy = np.abs(np.diff(y_trimmed, prepend=y_trimmed[0]))

    motion_mask = dy > 1e-4
    keep_mask = motion_mask.copy()

    # keep every 10th static sample
    keep_mask[~motion_mask] = np.arange(len(y_trimmed))[~motion_mask] % 30 == 0

    X_filtered = X_trimmed[keep_mask].reshape(-1,1)
    y_filtered = y_trimmed[keep_mask].reshape(-1,1)

    model = LinearRegression()
    model.fit(X_filtered, y_filtered)

    slope = model.coef_[0, 0]
    intercept = model.intercept_[0]

    print(f"\n--- Segment {seg['segment']} ---")
    print(f"slope: {slope:.4f}, intercept: {intercept:.4f}")

    calibration_models.append({
        "segment": seg["segment"],
        "model": model,
        "slope": slope,
        "intercept": intercept,
    })

    
    # ─────────────────────────────────────
    # Optional: plot per segment
    # ─────────────────────────────────────
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.figure()
    plt.scatter(X, y, s=2, alpha=0.4)
    plt.plot(x_line, y_line, color='orange')
    plt.xlabel("PCA")
    plt.ylabel("Position in cm")
    plt.title(f"Segment  {seg['segment']}: Slope = {slope:.3f}; Intercept = {intercept:.3f} ")
    
    filename = os.path.join(single_reg_dir, f"RegressionModel_segment{seg['segment']}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    
    
plot_calibration_models(calibration_models)

#TESTETSTESTESTETSTETSTETSTETSTESTESTET
# ─────────────────────────────────────
n_segments = len(segments)
n_calibration = len(calibration_models)
print(n_segments)
print(n_calibration)

calibration_matrix_rmse = np.zeros(
    (n_segments, n_segments)
)
calibration_matrix_mae = np.zeros(
    (n_segments, n_segments)
)
calibration_matrix_max_error = np.zeros(
    (n_segments, n_segments)
)




for i, calibration in enumerate(calibration_models):

    #cal_id = calibration["segment"]
    model = calibration["model"]


    for j, test in enumerate(segments):

        #test_id = test["segment"]
        print(f"Calibration {i} -> Test {j}")

        pc1_test = test["pc1"]


        # -------------------------------
        # Motor ground truth
        # -------------------------------        
        y_motor = test["motor_positions"] / 2000
        t_motor = test["motor_times"]
        t_radar = test["radar_times"]

        # -------------------------------
        # Prediction
        # -------------------------------
        y_pred = model.predict(pc1_test.reshape(-1,1))
        y_interp = np.interp( t_motor, t_radar, y_pred.squeeze())
        
        
        # -------------------------------
        # RMSE and Plots
        # -------------------------------
        error_rmse = rmse(y_motor, y_interp)
        error_mae = mae(y_motor, y_interp)
        error_max = max_abs_error(y_motor, y_interp)

        calibration_matrix_rmse[i, j] = error_rmse
        calibration_matrix_mae[i, j] = error_mae
        calibration_matrix_max_error[i, j] = error_max


        plt.figure(figsize=(10,5))

        plt.plot(t_motor, y_motor, label="True motor movement", linewidth=2)
        plt.plot(t_radar, y_pred, label="Predicted (PCA)", alpha=0.8, color='orange')
        plt.plot(t_motor, y_interp, label="Predicted (interp)", alpha=0.8, color='green')

        plt.xlabel("Time (s)")
        plt.ylabel("Motor displacement (cm)")

        plt.title(
            f"Cal {i} → Test {j} | RMSE = {error_rmse:.4f}"
        )

        plt.legend()
        plt.grid()


        # ----------------------------------
        # SAVE FIGURE
        # ----------------------------------
        #save_dir = "Data/calibration_plots"
        
        filename = os.path.join(cal_test_dir, f"cal_{i}_test_{j}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()


# ─────────────────────────────────────
# Plot combined movement for test segments 7–13
# ─────────────────────────────────────
aggregate_calibration_segment = 1
aggregate_model = next(
    calibration["model"]
    for calibration in calibration_models
    if calibration["segment"] == aggregate_calibration_segment
)

aggregate_times = []
aggregate_true = []
aggregate_predicted = []

for test in segments:
    if 7 <= test["segment"] <= 13:
        y_pred = aggregate_model.predict(test["pc1"].reshape(-1, 1)).squeeze()
        y_interp = np.interp(
            test["motor_times"],
            test["radar_times"],
            y_pred,
        )

        aggregate_times.append(test["motor_times"])
        aggregate_true.append(test["motor_positions"] / 2000)
        aggregate_predicted.append(y_interp)

if aggregate_times:
    aggregate_times = np.concatenate(aggregate_times)
    aggregate_true = np.concatenate(aggregate_true)
    aggregate_predicted = np.concatenate(aggregate_predicted)

    plt.figure(figsize=(12, 5))
    plt.plot(aggregate_times, aggregate_true, label="True motor movement", linewidth=2)
    plt.plot(
        aggregate_times,
        aggregate_predicted,
        label=f"Predicted movement (calibration segment {aggregate_calibration_segment})",
        color="orange",
        alpha=0.85,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Motor displacement (cm)")
    plt.title("Combined prediction for test segments 7–13")
    plt.legend()
    plt.grid()

    filename = os.path.join(cal_test_dir, "combined_segments_7_13.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()



# ─────────────────────────────────────
# Plot and save RMSE calibration matrix
# ─────────────────────────────────────
plt.figure(figsize=(7,6))

plt.imshow(
    calibration_matrix_rmse[:6, :],
    aspect="auto"
)
plt.colorbar(label="RMSE")
plt.xlabel("Test segment")
plt.ylabel("Calibration segment")
plt.title("Calibration Transfer Matrix - RMSE")

filename = os.path.join(cal_matrix_dir, "calibration_matrix_rmse.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
# wandb.log({
#     "calibration_matrix_rmse": wandb.Image(plt)
# })
plt.close()

# ─────────────────────────────────────
# Plot and save MAE calibration matrix
# ─────────────────────────────────────
plt.figure(figsize=(7,6))

plt.imshow(
    calibration_matrix_mae[:6, :],
    aspect="auto"
)
plt.colorbar(label="MAE")
plt.xlabel("Test segment")
plt.ylabel("Calibration segment")
plt.title("Calibration Transfer Matrix - MAE")

filename = os.path.join(cal_matrix_dir, "calibration_matrix_mae.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
# wandb.log({
#     "calibration_matrix_mae": wandb.Image(plt)
# })
plt.close()

# ─────────────────────────────────────
# Plot and save Max Error calibration matrix
# ─────────────────────────────────────
plt.figure(figsize=(7,6))

plt.imshow(
    calibration_matrix_max_error[:6, :],
    aspect="auto"
)
plt.colorbar(label="Max Abs Error")
plt.xlabel("Test segment")
plt.ylabel("Calibration segment")
plt.title("Calibration Transfer Matrix - Max Absolute Error")

filename = os.path.join(cal_matrix_dir, "calibration_matrix_max_error.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
# wandb.log({
#     "calibration_matrix_max_error": wandb.Image(plt)
# })
plt.close()

print("mean RMSE:", np.mean(calibration_matrix_rmse[:6, :]))
print("mean MAE:", np.mean(calibration_matrix_mae[:6, :]))
print("mean Max Error:", np.mean(calibration_matrix_max_error[:6, :]))

# ─────────────────────────────────────








