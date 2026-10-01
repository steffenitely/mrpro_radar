import json
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.linear_model import LinearRegression
from mrpro.operators.PCACompressionOp import PCACompressionOp
import torch
import os
import wandb

wandb.init(
    project="radar-motor-ApproachComparison-TorchModel",
    config={
        "segment_length": 40.0,
        "model": "LinearRegression",
        "pca_components": 510,
        "trim_ratio": 0.1
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE — File paths
# ══════════════════════════════════════════════════════════════════════════════
# Calibrate and rnd Movement
#Both done in one measurement
RADAR = "Data/RadarTest/radar_20260729_145247.npz"
MOTOR = "Data/TimeLogs/time_log_20260729_145342.json"  

save_dir = "Data/calibration_plots/ApproachComparison2/00"

motor_pca_dir = os.path.join(save_dir, "Motor_vs_PCA")
regression_dir = os.path.join(save_dir, "RegressionModels")
single_reg_dir = os.path.join(save_dir, "RegressionModel_per_segment")
cal_test_dir = os.path.join(save_dir, "Calibration_vs_Test")
cal_matrix_dir = os.path.join(save_dir, "Calibration_Matrix")

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

    #detrend and remove clutter from radar data
    rx1 = radar_cube[0, :, :]
    rx2 = radar_cube[1, :, :]
    rx1_cf = remove_clutter(rx1)
    rx2_cf = remove_clutter(rx2)
    rx1_cf = detrend(rx1_cf)
    rx2_cf = detrend(rx2_cf)
    radar_cube[0, :, :] = rx1_cf
    radar_cube[1, :, :] = rx2_cf
    
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
# PyTorch Model
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # <-- THIS is linear regression

    def forward(self, x):
        return self.linear(x)

def train_model(X, y, segment_id, epochs=1000, lr=1e-1):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotorNet().to(device)

    patience = 50
    counter = 0
    min_delta = 1e-6


    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    prev_loss = float("inf")  # or None

    for epoch in range(epochs):
        model.train()

        y_pred = model(X_t)
        loss = loss_fn(y_pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if prev_loss - loss.item() < min_delta:
            counter += 1
        else:
            counter = 0

        # Log to W&B
        if epoch % 50 == 0:
            wandb.log({
            f"train_loss_seg_{segment_id}": loss.item(),
            "epoch": epoch
            })

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            print(f"Final loss: {loss.item():.6f}")
            break

        prev_loss = loss.item()

    model.eval()

    return model

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

def align_radar(radar, peaks, ref_bin=200):
    T, R = radar.shape
    aligned = np.zeros_like(radar)

    for t in range(T):
        shift = ref_bin - peaks[t]
        aligned[t] = np.roll(radar[t], shift)

    return aligned

def find_peaks_per_frame(radar):
    # radar: (T, R)
    peaks = np.argmax(radar, axis=1)   # shape: (T,)
    return peaks


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

    return rx1, rx2

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
    wandb.log({"regression_models": wandb.Image(plt)})
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



# ==============================================================
# MAIN
# ==============================================================

segments = allign_and_split(RADAR, MOTOR, segment_length=40.0)

all_segments = []
reference_mean = None

# ==============================================================
# STEP 1:
# Prepare all segments and obtain reference mean from segment 0
# ==============================================================

for i in range(len(segments)):

    radar_cube_seg, radar_times, motor_times, motor_positions = segments[i]

    print(f"\n--- Segment {i} ---")

    # Skip empty segments
    if len(radar_times) == 0 or len(motor_times) < 2:
        print("Skipping (not enough data)")
        continue

    rx1, rx2, *_ = basic_info_extraction(
        radar_cube_seg
    )

    M_both = np.hstack([rx1, rx2])

    # Reference mean from segment 0
    if i == 0:
        reference_mean = np.mean(M_both, axis=0)

        print(
            "Stored reference mean from segment 0; "
            "skipping segment 0 for the rest of the analysis"
        )
        continue

    all_segments.append({
        "segment": i,
        "radar_cube": radar_cube_seg,
        "radar_times": radar_times,
        "motor_times": motor_times,
        "motor_positions": motor_positions,
        "M_both": M_both
    })


# ==============================================================
# STEP 2:
# Train ONLY model 3 using Segment -> PCA
# ==============================================================

calibration_segment_id = 3

calibration_segment = next(
    seg for seg in all_segments
    if seg["segment"] == calibration_segment_id
)

M_cal = calibration_segment["M_both"]

# Use reference mean
M_cal_centered = torch.tensor(
    M_cal - reference_mean,
    dtype=torch.float32
)

# PCA specifically on Segment 3
op_model3 = PCACompressionOp(
    data=M_cal_centered,
    n_components=510,
    centering=False
)

pca_model3 = op_model3(M_cal_centered)[0]

pc1_model3 = pca_model3[:, 0].numpy()


# --------------------------------------------------------------
# Align motor data for Segment 3
# --------------------------------------------------------------

radar_times_3 = calibration_segment["radar_times"]
motor_times_3 = calibration_segment["motor_times"]
motor_positions_3 = calibration_segment["motor_positions"]

y_aligned_3 = np.interp(
    radar_times_3,
    motor_times_3,
    motor_positions_3
)

y_3 = y_aligned_3 / 2000

X_3 = pc1_model3


# --------------------------------------------------------------
# Trim extremes
# --------------------------------------------------------------

xmin, xmax = X_3.min(), X_3.max()
span = xmax - xmin

lower = xmin + 0.1 * span
upper = xmax - 0.1 * span

mask = (X_3 >= lower) & (X_3 <= upper)

X_trimmed = X_3[mask].reshape(-1, 1)
y_trimmed = y_3[mask]


# --------------------------------------------------------------
# Downsample static regions
# --------------------------------------------------------------

dy = np.abs(
    np.diff(
        y_trimmed,
        prepend=y_trimmed[0]
    )
)

motion_mask = dy > 1e-4
keep_mask = motion_mask.copy()

# Keep every 30th static sample
keep_mask[~motion_mask] = (
    np.arange(len(y_trimmed))[~motion_mask] % 30 == 0
)

X_filtered = X_trimmed[keep_mask].reshape(-1, 1)
y_filtered = y_trimmed[keep_mask].reshape(-1, 1)


# --------------------------------------------------------------
# Train MODEL 3
# --------------------------------------------------------------

model_3 = train_model(
    X_filtered,
    y_filtered,
    calibration_segment_id
)

model_3.eval()

print("\n========================================")
print("MODEL 3 TRAINED")
print("========================================")

print(
    f"Slope:     {model_3.linear.weight.item():.4f}"
)

print(
    f"Intercept: {model_3.linear.bias.item():.4f}"
)


# ==============================================================
# STEP 3:
# SEGMENT -> PCA
#
# Calculate PCA independently for each segment 7-13
# and use MODEL 3 for prediction
# ==============================================================

segment_pca_results = []


for test in all_segments:

    test_id = test["segment"]

    if test_id < 7 or test_id > 13:
        continue

    print(
        f"Segment -> PCA: evaluating segment {test_id}"
    )

    # ----------------------------------------------------------
    # Extract radar data
    # ----------------------------------------------------------

    M_test = test["M_both"]

    M_test_centered = torch.tensor(
        M_test - reference_mean,
        dtype=torch.float32
    )

    # ----------------------------------------------------------
    # PCA independently for THIS segment
    # ----------------------------------------------------------

    op_test = PCACompressionOp(
        data=M_test_centered,
        n_components=510,
        centering=False
    )

    test_pca = op_test(M_test_centered)[0]

    pc1_test = test_pca[:, 0].numpy()

    # ----------------------------------------------------------
    # Predict using MODEL 3
    # ----------------------------------------------------------

    X_test_t = torch.tensor(
        pc1_test.reshape(-1, 1),
        dtype=torch.float32
    )

    with torch.no_grad():
        y_pred = (
            model_3(X_test_t)
            .squeeze()
            .cpu()
            .numpy()
        )

    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    t_motor = test["motor_times"]
    y_motor = test["motor_positions"] / 2000

    t_radar = test["radar_times"]

    y_interp = np.interp(
        t_motor,
        t_radar,
        y_pred
    )

    segment_pca_results.append({
        "segment": test_id,
        "t_motor": t_motor,
        "y_motor": y_motor,
        "t_radar": t_radar,
        "y_pred": y_pred,
        "y_interp": y_interp
    })


# ==============================================================
# Plot SEGMENT -> PCA
# ==============================================================

plt.figure(figsize=(12, 6))

for result in segment_pca_results:

    plt.plot(
        result["t_motor"],
        result["y_motor"],
        linewidth=2,
        label=f"True - Segment {result['segment']}"
    )

    plt.plot(
        result["t_motor"],
        result["y_interp"],
        alpha=0.8,
        label=f"Predicted - Segment {result['segment']}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Motor displacement (cm)")

plt.title(
    "Segment → PCA → Model 3 | Segments 7–13"
)

plt.legend(
    ncol=2,
    fontsize=8
)

plt.grid()

filename = os.path.join(
    cal_test_dir,
    "model3_segment_pca_segments7_13.png"
)

plt.savefig(
    filename,
    dpi=150,
    bbox_inches="tight"
)

wandb.log({
    "model3_segment_pca_segments7_13":
        wandb.Image(plt)
})

plt.show()
plt.close()


# ==============================================================
# STEP 4:
# PCA -> SEGMENT
#
# First calculate ONE PCA over ALL radar data.
# Then split the resulting PC1 into segments.
# ==============================================================

print("\n========================================")
print("PCA -> SEGMENT")
print("Calculating global PCA...")
print("========================================")


# --------------------------------------------------------------
# Concatenate all radar data
# --------------------------------------------------------------

M_all = np.vstack([
    seg["M_both"]
    for seg in all_segments
])

M_all_centered = torch.tensor(
    M_all - reference_mean,
    dtype=torch.float32
)


# --------------------------------------------------------------
# ONE global PCA
# --------------------------------------------------------------

# op_global = PCACompressionOp(
#     data=M_all_centered,
#     n_components=510,
#     centering=False
# )

pca_global = op_model3(M_all_centered)[0]

pc1_global = pca_global[:, 0].numpy()


# --------------------------------------------------------------
# Keep track of where each segment starts/ends
# --------------------------------------------------------------

segment_lengths = [
    len(seg["M_both"])
    for seg in all_segments
]

segment_boundaries = np.cumsum(
    [0] + segment_lengths
)


# ==============================================================
# Predict segments 7-13 using MODEL 3
# ==============================================================

global_pca_results = []


for k, test in enumerate(all_segments):

    test_id = test["segment"]

    if test_id < 7 or test_id > 13:
        continue

    print(
        f"PCA -> Segment: evaluating segment {test_id}"
    )

    start = segment_boundaries[k]
    end = segment_boundaries[k + 1]

    # ----------------------------------------------------------
    # Extract this segment from GLOBAL PC1
    # ----------------------------------------------------------

    pc1_test = pc1_global[start:end]

    # ----------------------------------------------------------
    # Predict using the SAME MODEL 3
    # ----------------------------------------------------------

    X_test_t = torch.tensor(
        pc1_test.reshape(-1, 1),
        dtype=torch.float32
    )

    with torch.no_grad():
        y_pred = (
            model_3(X_test_t)
            .squeeze()
            .cpu()
            .numpy()
        )

    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    t_motor = test["motor_times"]
    y_motor = test["motor_positions"] / 2000

    t_radar = test["radar_times"]

    y_interp = np.interp(
        t_motor,
        t_radar,
        y_pred
    )

    global_pca_results.append({
        "segment": test_id,
        "t_motor": t_motor,
        "y_motor": y_motor,
        "t_radar": t_radar,
        "y_pred": y_pred,
        "y_interp": y_interp
    })


# ==============================================================
# Plot PCA -> SEGMENT
# ==============================================================

plt.figure(figsize=(12, 6))

for result in global_pca_results:

    plt.plot(
        result["t_motor"],
        result["y_motor"],
        linewidth=2,
        label=f"True - Segment {result['segment']}"
    )

    plt.plot(
        result["t_motor"],
        result["y_interp"],
        alpha=0.8,
        label=f"Predicted - Segment {result['segment']}"
    )

plt.xlabel("Time (s)")
plt.ylabel("Motor displacement (cm)")

plt.title(
    "PCA → Segment → Model 3 | Segments 7–13"
)

plt.legend(
    ncol=2,
    fontsize=8
)

plt.grid()

filename = os.path.join(
    cal_test_dir,
    "model3_pca_segment_segments7_13.png"
)

plt.savefig(
    filename,
    dpi=150,
    bbox_inches="tight"
)

wandb.log({
    "model3_pca_segment_segments7_13":
        wandb.Image(plt)
})

plt.show()
plt.close()

# ==============================================================
# Combined plot:
# True vs Segment -> PCA vs PCA -> Segment
# ==============================================================

plt.figure(figsize=(14, 6))

# --------------------------------------------------------------
# Plot segments 7-13
# --------------------------------------------------------------

for seg_pca, global_pca in zip(
    segment_pca_results,
    global_pca_results
):

    segment_id = seg_pca["segment"]

    # ----------------------------------------------------------
    # True motor movement
    # ----------------------------------------------------------

    plt.plot(
        seg_pca["t_motor"],
        seg_pca["y_motor"],
        color="red",
        alpha=0.5,
        linestyle="--",   # dashed
        linewidth=2,
        label="True motor movement"
        if segment_id == 7 else None
    )

    # ----------------------------------------------------------
    # Segment -> PCA -> Model 3
    # ----------------------------------------------------------

    plt.plot(
        seg_pca["t_motor"],
        seg_pca["y_interp"],
        color="blue",
        alpha=0.8,
        linewidth=1.5,
        label="Segment → PCA → Model 3"
        if segment_id == 7 else None
    )

    # ----------------------------------------------------------
    # PCA -> Segment -> Model 3
    # ----------------------------------------------------------

    plt.plot(
        global_pca["t_motor"],
        global_pca["y_interp"],
        color="green",
        alpha=0.8,
        linewidth=1.5,
        label="PCA → Segment → Model 3"
        if segment_id == 7 else None
    )


# --------------------------------------------------------------
# Formatting
# --------------------------------------------------------------

plt.xlabel("Time (s)")
plt.ylabel("Motor displacement (cm)")
plt.xticks([280, 320, 360, 400, 440, 480, 520], fontsize=10)


plt.title(
    "Model 3: Segment → PCA vs PCA → Segment | Segments 7–13"
)

plt.legend()

plt.grid()

filename = os.path.join(
    cal_test_dir,
    "model3_segment_pca_vs_pca_segment_segments7_13.png"
)

plt.savefig(
    filename,
    dpi=150,
    bbox_inches="tight"
)

wandb.log({
    "model3_segment_pca_vs_pca_segment_segments7_13":
        wandb.Image(plt)
})

plt.show()
plt.close()
# ==============================================================
# Finish
# ==============================================================

wandb.config.update({
    "calibration_segment": 3,
    "test_segments": "6-12",
    "model_used_for_all_predictions": "segment_3",
    "comparison": "segment_pca_vs_pca_segment"
})

wandb.finish()