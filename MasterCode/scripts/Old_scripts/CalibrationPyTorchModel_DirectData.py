import json
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import datetime
import torch
import os
import wandb

wandb.init(
    project="radar-motor-direct-radar-data",
    config={
        "segment_length": 40.0,
        "model": "PyTorch MLP with direct segmented radar data",
        "input_type": "clutter_removed_detrended_radar",
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

save_dir = "Data/calibration_plots/SegmentationWandB_Nonlinear_RawData/00"


cal_test_dir = os.path.join(save_dir, "Calibration_vs_Test")
cal_matrix_dir = os.path.join(save_dir, "Calibration_Matrix")

for d in [cal_test_dir, cal_matrix_dir]:
    os.makedirs(d, exist_ok=True)
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_EPOCH: datetime.datetime | None = None
# ══════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# Load & parse the motor log
# ─────────────────────────────────────────────────────────────────────────────

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

def align_and_split(radar_path: str, motor_path: str, segment_length: float):
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
    rx1 = radar_cube[0, :, :]
    rx2 = radar_cube[1, :, :]
    rx1_cf = remove_clutter(rx1)
    rx2_cf = remove_clutter(rx2)
    rx1_cf = detrend(rx1_cf, axis=0)
    rx2_cf = detrend(rx2_cf, axis=0)
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
# PyTorch Model with Configurable Input Size
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class MotorNetRawData(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        
        # Dynamically scale hidden layers based on input size
        hidden1 = max(128, input_size // 2)
        hidden2 = max(64, input_size // 4)
        hidden3 = 32

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )

    def forward(self, x):
        return self.network(x)

def train_model(X, y, segment_id, epochs=5000, lr=1e-3):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get input size from data
    input_size = X.shape[1] if X.ndim > 1 else 1
    model = MotorNetRawData(input_size=input_size).to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    prev_loss = float("inf")  # or None

    patience = 200
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

    return model.cpu()

# ─────────────────────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────────────────────
def remove_clutter(radar_data_data):
    # radar_data: (T, R)
    return radar_data_data - np.mean(radar_data_data, axis=0, keepdims=True)

def highpass_clutter(radar_data):
    return detrend(radar_data, axis=0)


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


def plot_nonlinear_model(
    model: nn.Module,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str = "Raw Data",
    y_label: str = "Position in cm",
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot predictions against measured motor positions."""
    model.eval()

    x_values_t = torch.tensor(x_values, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_values_t).squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_values, predictions, s=8, alpha=0.4, color="#2c7fb8")
    limits = [
        min(float(np.min(y_values)), float(np.min(predictions))),
        max(float(np.max(y_values)), float(np.max(predictions))),
    ]
    ax.plot(limits, limits, color="#d95f02", linewidth=2)
    ax.set_xlabel("Measured position")
    ax.set_ylabel(y_label)
    ax.set_title(title or "Direct radar calibration model")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        base_name = "nonlinear_model_raw"
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

    plt.xlabel("Raw Data Component")
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


segments = align_and_split(RADAR, MOTOR, segment_length=40.0)

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

    rx1, rx2 = basic_info_extraction(radar_cube_seg)


    # ─────────────────────────────────────────────────────────────────────────────
    # Use raw radar data directly (both channels concatenated)
    # ─────────────────────────────────────────────────────────────────────────────
    M_both = np.hstack([rx1, rx2])

    if i == 0:
        reference_mean = np.mean(M_both, axis=0)
        print("Stored reference mean from segment 0; skipping segment 0 for the rest of the analysis")
        continue

    mean = reference_mean if reference_mean is not None else np.mean(M_both, axis=0)
    
    M_centered = M_both #- mean

    all_segments.append({
        "segment": i,
        "radar_cube": radar_cube_seg,
        "radar_times": radar_times,
        "motor_times": motor_times,
        "motor_positions": motor_positions
    })


    if i >= 7:
        print("Skipping model training for this segment")
        continue



    # ─────────────────────────────────────
    # Align motor
    # ─────────────────────────────────────
    y_aligned = np.interp(radar_times, motor_times, motor_positions)
    y = y_aligned / 2000

    X = M_centered  # Use raw data directly



    y = y.reshape(-1, 1)
    
    model = train_model(X, y, i)
    wandb.watch(model, log="all")
    model_path = os.path.join(save_dir, f"model_segment_{i}.pkl")


    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

    
    # Log training metrics
    wandb.log({
        "segment": i,
        "num_samples": len(X),
        "input_size": X.shape[1]
    })


    # store results
    calibration_models.append({
        "segment": i,
        "model": model,
        "mean": mean,
        "input_size": X.shape[1]
    })

    
    # ─────────────────────────────────────
    # plot per segment
    # ─────────────────────────────────────
    #filename = os.path.join(single_model_dir, f"RegressionModel_segment{i}.png")
    # fig, ax = plot_nonlinear_model(
    #     model=model,
    #     x_values=X,
    #     y_values=y,
    #     x_label="Raw Radar Data",
    #     y_label="Position in cm",
    #     title=f"Segment {i}: Nonlinear calibration model f(x)",
    #     save_path=filename,
    # )
    # wandb.log({
    #     f"regression_plot_segment_{i}": wandb.Image(fig)
    # })
    # plt.close(fig)
    
    
#plot_calibration_models(calibration_models)

# ─────────────────────────────────────
#calibration Matrix
# ─────────────────────────────────────
n_segments = len(all_segments)

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

    
    mean = calibration["mean"]
    model = calibration["model"]


    for j, test in enumerate(all_segments):

        test_id = test["segment"]
        print(f"Calibration {i} -> Test {j}")


        # -------------------------------
        # Extract radar features
        # -------------------------------
        rx1, rx2 = basic_info_extraction(test["radar_cube"])

        M_test = np.hstack([rx1, rx2])


        # -------------------------------
        # IMPORTANT:
        # use calibration mean
        # -------------------------------
        M_test_centered = M_test #- mean

        # ─────────────────────────────────────────────────────────────────────────────
        # Use raw radar data directly
        # ─────────────────────────────────────────────────────────────────────────────

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

        X_test_t = torch.tensor(M_test_centered, dtype=torch.float32)

        with torch.no_grad():
            y_pred = model(X_test_t).squeeze().cpu().numpy()

        y_interp = np.interp( t_motor, t_radar, y_pred)

        rmse_error = rmse(y_motor, y_interp)
        mae_error = mae(y_motor, y_interp)
        max_error = max_abs_error(y_motor, y_interp)
        
        wandb.log({
            f"prediction_plot_cal{i}_test{j}": wandb.Image(plt),
            "rmse": rmse_error,
            "mae": mae_error,
            "max_error": max_error
        })

        model_path = os.path.join(save_dir, f"model_segment_{i}.pt")
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # -------------------------------
        # Fill calibration matrices
        # -------------------------------
        
        calibration_matrix_rmse[i, j] = rmse_error
        calibration_matrix_mae[i, j] = mae_error
        calibration_matrix_max_error[i, j] = max_error


        plt.figure(figsize=(10,5))

        plt.plot(t_motor, y_motor, label="True motor movement", linewidth=2)
        plt.plot(t_radar, y_pred, label="Predicted (Raw Data)", alpha=0.8, color='orange')
        plt.plot(t_motor, y_interp, label="Predicted (interp)", alpha=0.8, color='green')

        plt.xlabel("Time (s)")
        plt.ylabel("Motor displacement (cm)")

        plt.title(
            f"Cal {i} → Test {j} | RMSE = {rmse_error:.4f}"
        )

        plt.legend()
        plt.grid()

        filename = os.path.join(cal_test_dir, f"cal_{i}_test_{j}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        wandb.log({
            f"prediction_plot_cal{i}_test{j}": wandb.Image(plt)
        })
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot and save RMSE calibration matrix
# ─────────────────────────────────────────────────────────────────────────────
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
wandb.log({
    "calibration_matrix_rmse": wandb.Image(plt)
})
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot and save MAE calibration matrix
# ─────────────────────────────────────────────────────────────────────────────
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
wandb.log({
    "calibration_matrix_mae": wandb.Image(plt)
})
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Plot and save Max Error calibration matrix
# ─────────────────────────────────────────────────────────────────────────────
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
wandb.log({
    "calibration_matrix_max_error": wandb.Image(plt)
})
plt.close()

wandb.config.update({
    "segments_used_for_calibration": 6
})
wandb.finish()
# ─────────────────────────────────────
