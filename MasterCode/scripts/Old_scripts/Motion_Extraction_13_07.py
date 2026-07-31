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

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE — File paths
# ══════════════════════════════════════════════════════════════════════════════
# Calibrate
RADAR_PATH = "Data/RadarTest/radar_20260618_114419.npz"
MOTOR_LOG_PATH = "Data/TimeLogs/time_log_20260618_114425.json"   # path to your JSON file

# Random Movement
RADAR_MOVEMENT_PREDICTION = "Data/RadarTest/radar_20260618_153139.npz"
MOVEMENT_LOG = "Data/TimeLogs/time_log_20260618_153144.json"

#Both done in one measurement
RADAR = "Data/RadarTest/radar_20260709_150948.npz"
MOTOR = "Data/TimeLogs/time_log_20260709_151058.json"  

RADAR2 = "Data/RadarTest/radar_20260707_150003.npz"
MOTOR2 = "Data/TimeLogs/time_log_20260707_150011.json"  

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE 1 — Action → physical position mapping
#   Add / rename keys to match every label that appears in your motor_log.
#   Values are the numeric position you want plotted (e.g. mm, degrees, …).
# ══════════════════════════════════════════════════════════════════════════════
ACTION_MAP: dict[str, float] = {
    "0": 0.0,    # e.g. home / reference position
    "A": 6000.0,    # e.g. first target position
    # "B": 2.0,
    # "C": 3.0,
}

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

def allign_and_split(radar_path: str, motor_path: str):
    radar_cube, timestamps, time_cube = load_radar(radar_path)
    print(radar_cube.shape)
    motor_times, motor_positions = load_motor_log(motor_path)

    # Radar time
    radar_datetimes = time_cube[0, :]
    radar_t_sec = radar_times_to_sec(radar_datetimes)

    # Align motor → radar timebase
    motor_t_aligned = align_motor_to_radar(
        motor_times, radar_datetimes, MOTOR_EPOCH
    )

    # -------------------
    # SPLIT TIME
    # -------------------
    split_time = 40.0  # seconds

    # Masks
    radar_mask_cal = radar_t_sec <= split_time
    radar_mask_rnd = radar_t_sec > split_time

    motor_mask_cal = motor_t_aligned <= split_time
    motor_mask_rnd = motor_t_aligned > split_time

    # Split radar
    radar_cube_cal = radar_cube[:, radar_mask_cal, :]
    radar_cube_rnd = radar_cube[:, radar_mask_rnd, :]

    radar_times_cal = radar_t_sec[radar_mask_cal]
    radar_times_rnd = radar_t_sec[radar_mask_rnd]

    # Split motor
    motor_times_cal = motor_t_aligned[motor_mask_cal]
    motor_positions_cal = motor_positions[motor_mask_cal]

    motor_times_rnd = motor_t_aligned[motor_mask_rnd]
    motor_positions_rnd = motor_positions[motor_mask_rnd]

    return (
        (radar_cube_cal, radar_times_cal, motor_times_cal, motor_positions_cal),
        (radar_cube_rnd, radar_times_rnd, motor_times_rnd, motor_positions_rnd),
    )

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
# Plot Motor vs. PCA
# ─────────────────────────────────────────────────────────────────────────────
def plot_pca_vs_motor(
    radar_times_sec: np.ndarray,
    pca_values: np.ndarray,          # shape (Timeframes,) — first PC or any scalar
    motor_times_aligned: np.ndarray,
    motor_positions: np.ndarray,
    pca_label: str = "PC 1",
    position_unit: str = "position",
) -> None:
    fig_M_vs_PCA, ax1 = plt.subplots(figsize=(14, 5))
    fig_M_vs_PCA.patch.set_facecolor("#f8f8f8")
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
    t_dense = np.linspace(motor_times_aligned[0], motor_times_aligned[-1], len(radar_times_sec))
    y_dense = np.interp(t_dense, motor_times_aligned, motor_positions)
    ax2.plot(t_dense, y_dense, color=color_motor, lw=2.2, label="Motor position", zorder=4, alpha=0.5)

    # mark each logged sample
    ax2.scatter(motor_times_aligned, motor_positions,
                color=color_motor, s=25, zorder=5)

    ax2.tick_params(axis="y", labelcolor=color_motor)

    # tidy y-ticks: one tick per known position, labelled with action name
    sorted_actions = sorted(ACTION_MAP.items(), key=lambda kv: kv[1])
    # ax2.set_yticks([v for _, v in sorted_actions])
    # ax2.set_yticklabels([k for k, _ in sorted_actions], fontsize=9)
    ax2.yaxis.set_major_locator(ticker.AutoLocator())

    # ── Legend ────────────────────────────────────────────────────────────────
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.85)

    plt.title("PCA  vs  Motor Position  (aligned to t = 0)", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig("pca_vs_motor.png", dpi=150, bbox_inches="tight")
    print("Saved → pca_vs_motor.png")
    plt.show()



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

def plot_pca_and_fft(pca_comps):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    axes[0].plot(pca_comps[:, 0], alpha=0.4); axes[0].plot(pca_comps[:, 1], alpha=0.4); axes[0].plot(pca_comps[:, 2], alpha=0.4);
    axes[0].plot(pca_comps[:, 3], alpha=0.4); axes[0].plot(pca_comps[:, 4], alpha=0.4); axes[0].plot(pca_comps[:, 5], alpha=0.4); 
    axes[0].set_title("PCs — motion signal"); axes[0].set_xlabel("Frame")
    freqs = np.fft.rfftfreq(len(pca_comps[:,0]), d=1/50)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 0])), alpha=0.4)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 1])), alpha=0.4)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 2])), alpha=0.4)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 3])), alpha=0.4)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 4])), alpha=0.4)
    axes[1].plot(freqs, np.abs(np.fft.rfft(pca_comps[:, 5])), alpha=0.4)
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_title("PCs frequency spectrum")
    #plt.tight_layout()

    plt.show()


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




#==============================================================MAIN===========================================


# motor_times_cal, motor_positions_cal = load_motor_log(MOTOR_LOG_PATH)
# motor_times_rnd, motor_positions_rnd = load_motor_log(MOVEMENT_LOG)

# radar_cube_cal, timeframes_cal, timecube_cal = load_radar(RADAR_PATH)
# radar_cube_rnd, timeframes_rnd, timecube_rnd = load_radar(RADAR_MOVEMENT_PREDICTION)



#comment if only one measurement
#────────────────────────────────────────────────────────────────────────────

(
    (radar_cube_cal, radar_times_cal, motor_times_cal, motor_positions_cal),
    (radar_cube_rnd, radar_times_rnd, motor_times_rnd, motor_positions_rnd),
) = allign_and_split(RADAR, MOTOR)

# (
#     (radar_cube_cal_poop, radar_times_cal_poop, motor_times_cal_poop, motor_positions_cal_poop),
#     (radar_cube_rnd, radar_times_rnd, motor_times_rnd, motor_positions_rnd),
# ) = allign_and_split(RADAR2, MOTOR2)

motor_t_aligned = motor_times_cal
radar_t_sec = radar_times_cal
#────────────────────────────────────────────────────────────────────────────


# radar_datetimes = timecube_cal[0,:]
# rnd_radar_datetimes = timecube_rnd[0,:]


# # ── Convert radar times → seconds from t=0 ───────────────────────────────
# radar_times_cal = radar_times_to_sec(radar_datetimes)
# radar_times_rnd = radar_times_to_sec(rnd_radar_datetimes)


# # ── Shift motor times to the same reference ───────────────────────────────
# motor_times_cal = align_motor_to_radar(
#     motor_times_cal, radar_datetimes, MOTOR_EPOCH
# )
# motor_times_rnd = align_motor_to_radar(
#     motor_times_rnd, rnd_radar_datetimes, MOTOR_EPOCH
# )




#basic plotting
rx1, rx2, rx1_cf, rx2_cf, peaks_rx1, peaks_rx2, energy1, energy2, rbin1, rbin2 = basic_info_extraction(radar_cube=radar_cube_cal)

#--- PCA with both antennas

M_both = np.hstack([rx1_cf, rx2_cf])  # (timeframes, 2*510)
mean = np.mean(M_both, axis=0)
M_train_centered = M_both - mean
M_train_centered = torch.tensor(M_train_centered, dtype=torch.float32)  # or float if real

op = PCACompressionOp(
    data=M_train_centered,
    n_components=510,
    centering=False
)

train_pca = op(M_train_centered)[0]
train_pc1 = train_pca[:, 0].numpy()   # (T,)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrating Model via linear regression
# ─────────────────────────────────────────────────────────────────────────────
t_dense = np.linspace(motor_times_cal[0], motor_times_cal[-1], len(radar_times_cal))
y_dense = np.interp(t_dense, motor_times_cal, motor_positions_cal)
y_aligned = np.interp(radar_times_cal, motor_times_cal, motor_positions_cal)


# 2. Build regression dataset
X = train_pc1        # shape (2765,)
plt.plot(X)
plt.show()
X_shaped = train_pc1.reshape(-1, 1)

y = y_aligned              # shape (2765,)
y = y/2000

#### discard most extreme values
xmin = X.min()
xmax = X.max()
span = xmax - xmin

lower = xmin + 0.1 * span
upper = xmax - 0.1 * span
mask = (X >= lower) & (X <= upper)



X_trimmed = X[mask].reshape(-1, 1)
y_trimmed = y[mask]



model = LinearRegression()
model.fit(X_trimmed, y_trimmed)

x_line = np.linspace(X_shaped.min(), X_shaped.max(), 100)
y_line = model.predict(x_line.reshape(-1, 1))


print("slope:", model.coef_[0])
print("intercept:", model.intercept_)


plt.scatter(X, y, s=2, alpha=0.4)
plt.plot(x_line, y_line, color='orange')
plt.ylabel("Verschiebung in cm")
plt.xlabel("PCA Data")
plt.title(f"Linear Regression - slope: {model.coef_[0]:.3f} ; intercept: {model.intercept_:.3f}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Predict Random Motor Movement via Model
# ─────────────────────────────────────────────────────────────────────────────
plot_motor(motor_times_rnd, motor_positions_rnd)

rx1, rx2, rx1_cf, rx2_cf, peaks_rx1, peaks_rx2, energy1, energy2, rbin1, rbin2 = basic_info_extraction(radar_cube=radar_cube_rnd)
M_both_new = np.hstack([rx1_cf, rx2_cf])  # (timeframes, 2*510)

#--- substract  calibraton mean from matrix
M_centered = M_both_new - mean #mean #np.mean(M_both_new, axis=0)#
M_centered = torch.tensor(M_centered, dtype=torch.float32)

new_pca = op(M_centered)[0]
pc1 = new_pca[:, 0].numpy()   # (T,)
pc1 = pc1.reshape(-1, 1)
plt.plot(pc1)
plt.title("pc1 of rnd movement data")
plt.show()


y_pred = model.predict(pc1)





# predict motor movement

print(f"y_pred shape: {y_pred.shape}")
y_pred = y_pred#*-1 +3
plt.plot(y_pred)
plt.title("y_predited")
plt.show()


t_motor = motor_times_rnd
y_motor = motor_positions_rnd / 2000


t_pred = np.linspace(t_motor[0], t_motor[-1], len(y_pred))
t_pred = radar_times_rnd 
print(f"t_pred shape: {t_pred.shape}")

#
y_pred_interp = np.interp(t_motor, t_pred, y_pred)
error = rmse(y_motor, y_pred_interp)
#print(error)

plt.figure(figsize=(10,5))
plt.plot(t_motor, y_motor, label="True motor movement", linewidth=2)
plt.plot(t_pred, y_pred, label="Predicted from radar PCA", alpha=0.8, color='orange')
plt.plot(t_motor, y_pred_interp, label="interp", alpha=0.8, color='green')
plt.xlabel("Time (s)")
plt.ylabel("Motor displacement (cm)")
plt.title(f"Motor reconstruction; RMSE = {error}")
plt.legend()
plt.grid()
plt.show()





# # ─────────────────────────────────────────────────────────────────────────────graveyard─────────────────────────────────────────────────────────────────────────────

## only try with max values so only a 2x2 matrix for model
# max_pca = max(pca_comps_cut[:,0])
# min_pca = min(pca_comps_cut[:,0])
# max_y = max(y)
# min_y = min(y)
# pca_extremes = np.array([max_pca, min_pca])
# X_xtreme = pca_extremes
# y_extremes = np.array([max_y, min_y])

# print("Regression matrix X shape:", X.shape)
# print("Xtreme matrix X shape:", X_xtreme.shape)

# model_xtreme = LinearRegression()
# model_xtreme.fit(pca_extremes.reshape(-1, 1), y_extremes)

# x_line_xtreme = np.linspace(pca_extremes.min(), pca_extremes.max(), 100)
# y_line_xtreme = model_xtreme.predict(x_line_xtreme.reshape(-1, 1))

#print("slope:", model_xtreme.coef_[0])
#print("intercept:", model_xtreme.intercept_)

#y_pred_xtreme = model_xtreme.predict(X_new)*-1+2
#plt.plot(t_pred, y_pred_xtreme, label="Predicted from Extremes", color='red', alpha=0.8)
