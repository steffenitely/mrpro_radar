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
    project="radar-motor-nonlinear-TorchModel",
    config={
        "segment_length": 40.0,
        "model": "PyTorch Nonlinear Regression",
        "pca_components": 510,
        "trim_ratio": 0.03
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE — File paths
# ══════════════════════════════════════════════════════════════════════════════
# Calibrate and rnd Movement
#Both done in one measurement
RADAR = "Data/RadarTest/radar_20260729_145247.npz"
MOTOR = "Data/TimeLogs/time_log_20260729_145342.json"  

save_dir = "Data/calibration_plots/SegmentationWandB_Nonlinear/00"

motor_pca_dir = os.path.join(save_dir, "Motor_vs_PCA")
regression_dir = os.path.join(save_dir, "RegressionModels")
single_model_dir = os.path.join(save_dir, "RegressionModel_per_segment")
cal_test_dir = os.path.join(save_dir, "Calibration_vs_Test")
cal_matrix_dir = os.path.join(save_dir, "Calibration_Matrix")

for d in [motor_pca_dir, regression_dir, single_model_dir, cal_test_dir, cal_matrix_dir]:
    os.makedirs(d, exist_ok=True)
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_EPOCH: datetime.datetime | None = None
# ══════════════════════════════════


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
    plt.savefig(os.path.join(motor_pca_dir, "pca_vs_motor.png"), dpi=150, bbox_inches="tight")
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

        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        return self.network(x)

def train_model(X, y, segment_id, epochs=2000, lr=1e-3):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MotorNet().to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    prev_loss = float("inf")  # or None

    patience = 50
    counter = 0
    min_delta = 1e-7


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


def plot_nonlinear_model(
    model: nn.Module,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str = "PCA",
    y_label: str = "Position in cm",
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot the learned nonlinear calibration function f(x) over a dense grid."""
    model.eval()

    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    x_grid = np.linspace(x_min, x_max, 300).astype(np.float32)
    x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        y_grid = model(x_grid_t).squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=8, alpha=0.4, color="#2c7fb8", label="Data")
    ax.plot(x_grid, y_grid, color="#d95f02", linewidth=2, label=r"$f(x)$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or "Nonlinear calibration model")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        base_name = "nonlinear_model"
        if title is not None:
            safe_title = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in title)
            base_name = safe_title.strip("._-") or base_name
        save_path = os.path.join(single_model_dir, f"{base_name}.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


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



#==============================================================MAIN===========================================
#────────────────────────────────────────────────────────────────────────────


segments = allign_and_split(RADAR, MOTOR, segment_length=40.0)

calibration_models = []
all_segments = []
reference_mean = None

for i in range(len(segments)):

    radar_cube_seg, radar_times, motor_times, motor_positions = segments[i]
    print(f"\n--- Segment {i} ---")

    # skip empty segments
    if len(radar_times) == 0 or len(motor_times) < 2:
        print("Skipping (not enough data)")
        continue

    rx1, rx2, rx1_cf, rx2_cf, *_ = basic_info_extraction(radar_cube_seg)


    M_both = np.hstack([rx1_cf, rx2_cf])

    if i == 0:
        reference_mean = np.mean(M_both, axis=0)
        print("Stored reference mean from segment 0; skipping segment 0 for the rest of the analysis")
        continue

    mean = reference_mean if reference_mean is not None else np.mean(M_both, axis=0)
    
    M_centered = torch.tensor(M_both - mean, dtype=torch.float32)

    op = PCACompressionOp(
        data=M_centered,
        n_components=510,
        centering=False
    )

    pca = op(M_centered)[0]
    pc1 = pca[:, 0].numpy()

    # plot_motor_vs_pca(pca_values=pc1,
    #                   motor_times_aligned=motor_times,
    #                   motor_positions=motor_positions,
    #                   radar_times_sec=radar_times,
    #                   segment=i
    #                   )

    all_segments.append({
        "segment": i,
        "radar_cube": radar_cube_seg,
        "radar_times": radar_times,
        "motor_times": motor_times,
        "motor_positions": motor_positions
    })


    if i >= 7:
        print("Skipping PCA/model for this segment")
        continue



    # ─────────────────────────────────────
    # Align motor
    # ─────────────────────────────────────
    y_aligned = np.interp(radar_times, motor_times, motor_positions)
    y = y_aligned / 2000

    X = pc1

    # ─────────────────────────────────────
    # Trim extremes
    # ─────────────────────────────────────
    xmin, xmax = X.min(), X.max()
    span = xmax - xmin

    lower = xmin + 0.03 * span
    upper = xmax - 0.03 * span

    mask = (X >= lower) & (X <= upper)

    X_trimmed = X[mask].reshape(-1, 1)
    y_trimmed = y[mask]

    # ─────────────────────────────────────
    # Train model
    # ─────────────────────────────────────



    X_trimmed = X[mask].reshape(-1,1)
    y_trimmed = y[mask].reshape(-1,1)


    model = train_model(X_trimmed, y_trimmed, i)
    wandb.watch(model, log="all")
    model_path = os.path.join(save_dir, f"model_segment_{i}.pkl")


    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    
    # Log training metrics
    wandb.log({
        "segment": i,
        "num_samples": len(X_trimmed)
    })


    # store results
    calibration_models.append({
        "segment": i,
        "model": model,
        "mean": mean, 
        "pca_operator": op  
    })

    
    # ─────────────────────────────────────
    # plot per segment
    # ─────────────────────────────────────
    filename = os.path.join(single_model_dir, f"RegressionModel_segment{i}.png")
    fig, ax = plot_nonlinear_model(
        model=model,
        x_values=X,
        y_values=y,
        x_label="PCA",
        y_label="Position in cm",
        title=f"Segment {i}: Nonlinear calibration model f(x)",
        save_path=filename,
    )
    wandb.log({
        f"regression_plot_segment_{i}": wandb.Image(fig)
    })
    plt.close(fig)
    
    
#plot_calibration_models(calibration_models)

# ─────────────────────────────────────
#calibration Matrix
# ─────────────────────────────────────
n_segments = len(all_segments)

calibration_matrix = np.zeros(
    (n_segments, n_segments)
)


for i, calibration in enumerate(calibration_models):

    
    mean = calibration["mean"]
    op = calibration["pca_operator"]
    model = calibration["model"]


    for j, test in enumerate(all_segments):

        test_id = test["segment"]
        print(f"Calibration {i} -> Test {j}")


        # -------------------------------
        # Extract radar features
        # -------------------------------
        rx1, rx2, rx1_cf, rx2_cf, *_ = basic_info_extraction(test["radar_cube"])

        M_test = np.hstack([rx1_cf, rx2_cf])


        # -------------------------------
        # IMPORTANT:
        # use calibration mean
        # -------------------------------
        M_test_centered = M_test - mean
        M_test_centered = torch.tensor(
            M_test_centered,
            dtype=torch.float32
        )


        # -------------------------------
        # IMPORTANT:
        # use calibration PCA
        # -------------------------------
        test_pca = op(M_test_centered)[0]
        pc1_test = test_pca[:,0].numpy()

        # -------------------------------
        # Motor ground truth
        # -------------------------------        
        y_motor = test["motor_positions"] / 2000
        t_motor = test["motor_times"]
        t_radar = test["radar_times"]

        # -------------------------------
        # Prediction
        # -------------------------------
        model.eval()

        X_test_t = torch.tensor(pc1_test.reshape(-1,1), dtype=torch.float32)

        with torch.no_grad():
            y_pred = model(X_test_t).squeeze().cpu().numpy()

        y_interp = np.interp( t_motor, t_radar, y_pred)

        error = rmse(y_motor, y_interp)
        
        wandb.log({
            f"prediction_plot_cal{i}_test{j}": wandb.Image(plt),
            "rmse": error
        })

        model_path = os.path.join(save_dir, f"model_segment_{i}.pt")
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # -------------------------------
        # RMSE and Plots
        # -------------------------------
        
        calibration_matrix[i, j] = error


        plt.figure(figsize=(10,5))

        plt.plot(t_motor, y_motor, label="True motor movement", linewidth=2)
        plt.plot(t_radar, y_pred, label="Predicted (PCA)", alpha=0.8, color='orange')
        plt.plot(t_motor, y_interp, label="Predicted (interp)", alpha=0.8, color='green')

        plt.xlabel("Time (s)")
        plt.ylabel("Motor displacement (cm)")

        plt.title(
            f"Cal {i} → Test {j} | RMSE = {error:.4f}"
        )

        plt.legend()
        plt.grid()

        filename = os.path.join(cal_test_dir, f"cal_{i}_test_{j}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        wandb.log({
            f"prediction_plot_cal{i}_test{j}": wandb.Image(plt)
        })
        plt.close()


plt.figure(figsize=(7,6))

plt.imshow(
    calibration_matrix[:6, :],
    aspect="auto"
)
plt.colorbar(label="RMSE")
plt.xlabel("Test segment")
plt.ylabel("Calibration segment")
plt.title("Calibration Transfer Matrix")

filename = os.path.join(cal_matrix_dir, "calibration_matrix.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
wandb.log({
    "calibration_matrix": wandb.Image(plt)
})
plt.close()

wandb.config.update({
    "segments_used_for_calibration": 6
})
wandb.finish()
# ─────────────────────────────────────


