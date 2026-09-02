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

save_dir = "Data/calibration_plots/TrainValidateTest/00"

validation_dir = os.path.join(save_dir, "Validation")
test_dir = os.path.join(save_dir, "Test")


for d in [validation_dir, test_dir]:
    os.makedirs(d, exist_ok=True)
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_EPOCH: datetime.datetime | None = None
# ══════════════════════════════════════════════════════════════════════════════



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
# Split into Train/Validation/Test segments
# ─────────────────────────────────────────────────────────────────────────────
def prepare_segments(
    segments,
    reference_segment=0,
    ):
    """
    Prepare radar and motor data for train/validation/test.

    Returns:
        prepared_segments
        reference_mean
    """

    prepared_segments = []

    # --------------------------------------------------
    # Determine reference mean
    # --------------------------------------------------

    radar_cube_ref, radar_times_ref, motor_times_ref, motor_positions_ref = \
        segments[reference_segment]

    rx1_ref, rx2_ref = basic_info_extraction(radar_cube_ref)

    M_ref = np.hstack([rx1_ref, rx2_ref])

    reference_mean = np.mean(M_ref, axis=0)

    # --------------------------------------------------
    # Prepare all segments
    # --------------------------------------------------

    for i, segment in enumerate(segments):

        radar_cube, radar_times, motor_times, motor_positions = segment

        if len(radar_times) == 0 or len(motor_times) < 2:
            continue

        rx1, rx2 = basic_info_extraction(radar_cube)

        M_both = np.hstack([rx1, rx2])

        M_centered = M_both - reference_mean

        y_aligned = np.interp(
            radar_times,
            motor_times,
            motor_positions
        )

        y = y_aligned / 2000

        prepared_segments.append({
            "segment": i,
            "radar_cube": radar_cube,
            "radar_times": radar_times,
            "motor_times": motor_times,
            "motor_positions": motor_positions,

            "X": M_centered,
            "y": y,
        })

    return prepared_segments, reference_mean

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

    def train_model_with_early_stopping(
        X,
        y,
        validation_X=None,
        validation_y=None,
        segment_id=None,
        epochs=5000,
        lr=1e-3,
    ):
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = X.shape[1] if X.ndim > 1 else 1
        model = MotorNetRawData(input_size=input_size).to(device)

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        val_X_t = (
            torch.tensor(validation_X, dtype=torch.float32).to(device)
            if validation_X is not None
            else None
        )
        val_y_t = (
            torch.tensor(validation_y, dtype=torch.float32).to(device)
            if validation_y is not None
            else None
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        patience = 200
        counter = 0
        min_delta = 1e-7
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0

        for epoch in range(epochs):
            model.train()

            y_pred = model(X_t)
            train_loss = loss_fn(y_pred, y_t)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            val_loss = None
            if val_X_t is not None and val_y_t is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_X_t)
                    val_loss = loss_fn(val_pred, val_y_t).item()
                model.train()

            if val_loss is not None:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
            else:
                if best_val_loss - train_loss.item() > min_delta:
                    best_val_loss = train_loss.item()
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1

            if epoch % 10 == 0:
                wandb.log({
                    f"train_loss_{segment_id if segment_id is not None else 'model'}": train_loss.item(),
                    "epoch": epoch,
                    **({"validation_loss": val_loss} if val_loss is not None else {}),
                })

            if counter >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        return model.cpu()


    def train_model(
        X,
        y,
        validation_X=None,
        validation_y=None,
        segment_id=None,
        epochs=5000,
        lr=1e-3,
        ):
        torch.manual_seed(0)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Determine input size
        input_size = X.shape[1] if X.ndim > 1 else 1

        # Create model
        model = MotorNetRawData(input_size=input_size).to(device)

        # Convert training data to tensors
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        # Convert validation data if provided
        val_X_t = (
            torch.tensor(validation_X, dtype=torch.float32).to(device)
            if validation_X is not None
            else None
        )

        val_y_t = (
            torch.tensor(validation_y, dtype=torch.float32).to(device)
            if validation_y is not None
            else None
        )

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Keep track of the best validation model
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        for epoch in range(epochs):

            model.train()

            # Forward pass
            y_pred = model(X_t)

            # Training loss
            train_loss = loss_fn(y_pred, y_t)

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            val_loss = None

            if val_X_t is not None and val_y_t is not None:

                model.eval()

                with torch.no_grad():
                    val_pred = model(val_X_t)
                    val_loss = loss_fn(
                        val_pred,
                        val_y_t
                    ).item()

                # Check if this is the best model so far
                if val_loss < best_val_loss:

                    best_val_loss = val_loss
                    best_epoch = epoch

                    # Save a copy of the model weights
                    best_state = {
                        k: v.detach().clone()
                        for k, v in model.state_dict().items()
                    }

            # --------------------------------------------------
            # WandB logging
            # --------------------------------------------------

            if epoch % 20 == 0:

                log_data = {
                    f"train_loss_{segment_id if segment_id is not None else 'model'}":
                        train_loss.item(),
                    "epoch": epoch,
                }

                if val_loss is not None:
                    log_data["validation_loss"] = val_loss

                wandb.log(log_data)

        # --------------------------------------------------
        # Restore the model with the lowest validation loss
        # --------------------------------------------------

        if best_state is not None:

            model.load_state_dict(best_state)

            print(f"Training finished after {epochs} epochs")
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_val_loss:.6f}")

        else:

            print(f"Training finished after {epochs} epochs")
            print("No validation data was provided.")

        # Return model in evaluation mode on CPU
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



# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_SEGMENTS = [1, 2, 3, 4]
VALIDATION_SEGMENTS = [5, 6]
TEST_SEGMENTS = [7, 8, 9, 10, 11, 12, 13]
REFERENCE_SEGMENT = 0

# Keep the old names around for compatibility with the rest of the code
# while the actual model training uses train/val/test split semantics.
CALIBRATION_SEGMENT = TRAIN_SEGMENTS


# ============================================================
# 1. LOAD AND SPLIT DATA
# ============================================================

segments = align_and_split(
    RADAR,
    MOTOR,
    segment_length=40.0
)


# ============================================================
# 2. PREPARE DATA
# ============================================================

prepared_segments = []
reference_mean = None

for i, segment in enumerate(segments):

    radar_cube_seg, radar_times, motor_times, motor_positions = segment

    print(f"\n--- Segment {i} ---")

    # --------------------------------------------------------
    # Skip empty / invalid segments
    # --------------------------------------------------------

    if len(radar_times) == 0 or len(motor_times) < 2:
        print("Skipping: not enough data")
        continue

    # --------------------------------------------------------
    # Extract radar data
    # --------------------------------------------------------

    rx1, rx2 = basic_info_extraction(radar_cube_seg)
    M_both = np.hstack([rx1, rx2])

    # ========================================================
    # REFERENCE SEGMENT
    # ========================================================

    if i == REFERENCE_SEGMENT:

        reference_mean = np.mean(
            M_both,
            axis=0
        )

        print(
            "Stored reference mean from "
            f"segment {REFERENCE_SEGMENT}"
        )

        continue

    # --------------------------------------------------------
    # Make sure reference mean exists
    # --------------------------------------------------------

    if reference_mean is None:
        raise RuntimeError(
            "Reference mean has not been calculated."
        )

    # --------------------------------------------------------
    # Apply SAME reference mean to every segment
    # --------------------------------------------------------

    M_centered = M_both - reference_mean


    # --------------------------------------------------------
    # Align motor position to radar timestamps
    # --------------------------------------------------------

    y_aligned = np.interp(
        radar_times,
        motor_times,
        motor_positions
    )

    y = y_aligned / 2000.0


    # --------------------------------------------------------
    # Store prepared segment
    # --------------------------------------------------------

    prepared_segments.append({

        "segment": i,
        "radar_cube": radar_cube_seg,
        "radar_times": radar_times,
        "motor_times": motor_times,
        "motor_positions": motor_positions,
        "X": M_centered,
        "y": y
    })


print("\n==============================================")
print("Prepared segments:")
print([seg["segment"] for seg in prepared_segments])
print("==============================================")


# ============================================================
# 3. CREATE TRAIN / VALIDATION / TEST SETS
# ============================================================

def stack_dataset(dataset):
    X = np.vstack([seg["X"] for seg in dataset])
    y = np.concatenate([seg["y"] for seg in dataset]).reshape(-1, 1)
    return X, y


train_data = [
    seg
    for seg in prepared_segments
    if seg["segment"] in TRAIN_SEGMENTS
]

validation_data = [
    seg
    for seg in prepared_segments
    if seg["segment"] in VALIDATION_SEGMENTS
]

test_data = [
    seg
    for seg in prepared_segments
    if seg["segment"] in TEST_SEGMENTS
]


print("\n==============================================")
print("DATA SPLIT")
print("==============================================")

print("Training:", [seg["segment"] for seg in train_data])
print("Validation:", [seg["segment"] for seg in validation_data])
print("Test:", [seg["segment"] for seg in test_data])


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n==============================================")
print(f"TRAINING MODEL ON TRAIN SEGMENTS {TRAIN_SEGMENTS}")
print("==============================================")

X_train, y_train = stack_dataset(train_data)
X_val, y_val = stack_dataset(validation_data)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

model = MotorNetRawData.train_model(
    X_train,
    y_train,
    validation_X=X_val,
    validation_y=y_val,
    segment_id="train_val_split",
)


# ------------------------------------------------------------
# Save model ONCE
# ------------------------------------------------------------

model_path = os.path.join(
    save_dir,
    f"model_train_{TRAIN_SEGMENTS}_val_{VALIDATION_SEGMENTS}.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

wandb.save(model_path)

wandb.watch(
    model,
    log="all"
)


wandb.log({
    "train_segments": TRAIN_SEGMENTS,
    "validation_segments": VALIDATION_SEGMENTS,
    "test_segments": TEST_SEGMENTS,
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "input_size": X_train.shape[1],
})


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    data,
    dataset_name,
    output_dir
):

    model.eval()

    results = []


    for segment in data:

        segment_id = segment["segment"]
        X = segment["X"]
        t_radar = segment["radar_times"]
        t_motor = segment["motor_times"]
        y_motor = (
            segment["motor_positions"]
            / 2000.0
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )


        with torch.no_grad():

            y_pred = (
                model(X_tensor)
                .squeeze()
                .cpu()
                .numpy()
            )


        # ----------------------------------------------------
        # Interpolate prediction to motor timestamps
        # ----------------------------------------------------

        y_interp = np.interp(
            t_motor,
            t_radar,
            y_pred
        )


        # ----------------------------------------------------
        # Calculate errors
        # ----------------------------------------------------

        rmse_error = rmse(
            y_motor,
            y_interp
        )

        mae_error = mae(
            y_motor,
            y_interp
        )

        max_error = max_abs_error(
            y_motor,
            y_interp
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "segment": segment_id,
            "rmse": rmse_error,
            "mae": mae_error,
            "max_error": max_error
        })


        # ====================================================
        # PLOT
        # ====================================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            t_motor,
            y_motor,
            label="True motor movement",
            linewidth=2
        )

        plt.plot(
            t_radar,
            y_pred,
            label="Predicted",
            alpha=0.8
        )

        plt.plot(
            t_motor,
            y_interp,
            label="Predicted (interp)",
            alpha=0.8
        )

        plt.xlabel("Time (s)")
        plt.ylabel(
            "Motor displacement (normalized)"
        )
        plt.title(
            f"{dataset_name} | "
            f"Calibration S{CALIBRATION_SEGMENT} "
            f"→ Test S{segment_id}\n"
            f"RMSE = {rmse_error:.4f} | "
            f"MAE = {mae_error:.4f} | "
            f"Max Error = {max_error:.4f}"
        )
        plt.legend()
        plt.grid()


        # ----------------------------------------------------
        # Save plot
        # ----------------------------------------------------

        filename = os.path.join(
            output_dir,
            f"{dataset_name.lower()}_"
            f"cal_{CALIBRATION_SEGMENT}_"
            f"segment_{segment_id}.png"
        )


        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )


        # ----------------------------------------------------
        # WandB
        # ----------------------------------------------------

        wandb.log({

            f"{dataset_name}/"
            f"segment_{segment_id}/plot":
                wandb.Image(plt),

            f"{dataset_name}/"
            f"segment_{segment_id}/rmse":
                rmse_error,

            f"{dataset_name}/"
            f"segment_{segment_id}/mae":
                mae_error,

            f"{dataset_name}/"
            f"segment_{segment_id}/max_error":
                max_error
        })


        plt.close()


    return results


# ============================================================
# 6. VALIDATION
# ============================================================

print("\n==============================================")
print("VALIDATION")
print("==============================================")


validation_results = evaluate_model(
    model=model,
    data=validation_data,
    dataset_name="Validation",
    output_dir=validation_dir
)


# ------------------------------------------------------------
# Print validation results
# ------------------------------------------------------------

print("\nValidation results:")

for result in validation_results:

    print(
        f"Segment {result['segment']}: "
        f"RMSE = {result['rmse']:.4f}, "
        f"MAE = {result['mae']:.4f}, "
        f"Max = {result['max_error']:.4f}"
    )


# ============================================================
# 7. FINAL TEST
# ============================================================

print("\n==============================================")
print("FINAL TEST")
print("==============================================")


test_results = evaluate_model(
    model=model,
    data=test_data,
    dataset_name="Test",
    output_dir=test_dir
)


# ------------------------------------------------------------
# Print test results
# ------------------------------------------------------------

print("\nTest results:")

for result in test_results:

    print(
        f"Segment {result['segment']}: "
        f"RMSE = {result['rmse']:.4f}, "
        f"MAE = {result['mae']:.4f}, "
        f"Max = {result['max_error']:.4f}"
    )


# ============================================================
# 8. TEST RESULTS AS ARRAYS
# ============================================================

test_rmse = np.array([
    result["rmse"]
    for result in test_results
])

test_mae = np.array([
    result["mae"]
    for result in test_results
])

test_max_error = np.array([
    result["max_error"]
    for result in test_results
])


# ============================================================
# 9. OVERALL TEST PERFORMANCE
# ============================================================

mean_test_rmse = np.mean(test_rmse)

mean_test_mae = np.mean(test_mae)

mean_test_max_error = np.mean(
    test_max_error
)


print("\n==============================================")
print("FINAL TEST PERFORMANCE")
print("==============================================")

print(
    f"Mean RMSE      = {mean_test_rmse:.4f}"
)

print(
    f"Mean MAE       = {mean_test_mae:.4f}"
)

print(
    f"Mean Max Error = {mean_test_max_error:.4f}"
)


# ============================================================
# 10. LOG OVERALL TEST PERFORMANCE
# ============================================================

wandb.log({

    "final_test/mean_rmse":
        mean_test_rmse,

    "final_test/mean_mae":
        mean_test_mae,

    "final_test/mean_max_error":
        mean_test_max_error
})


# ============================================================
# 11. TEST RMSE MATRIX
# ============================================================

# One calibration model (segment 3)
# evaluated on all test segments.

test_rmse_matrix = np.array([
    test_rmse
])


plt.figure(
    figsize=(10, 3)
)


plt.imshow(
    test_rmse_matrix,
    aspect="auto"
)


plt.colorbar(
    label="RMSE"
)


plt.xticks(
    range(len(TEST_SEGMENTS)),
    TEST_SEGMENTS
)


plt.yticks(
    [0],
    [f"Calibration S{CALIBRATION_SEGMENT}"]
)


plt.xlabel("Test segment")

plt.ylabel("Calibration model")

plt.title(
    "Final Test Performance - RMSE"
)


filename = os.path.join(
    test_dir,
    "final_test_rmse_matrix.png"
)


plt.savefig(
    filename,
    dpi=150,
    bbox_inches="tight"
)


wandb.log({
    "final_test/rmse_matrix":
        wandb.Image(plt)
})


plt.close()


# ============================================================
# 12. FINISH
# ============================================================

wandb.config.update({

    "reference_segment":
        REFERENCE_SEGMENT,

    "train_segments":
        TRAIN_SEGMENTS,

    "validation_segments":
        VALIDATION_SEGMENTS,

    "test_segments":
        TEST_SEGMENTS,

    "calibration_segment":
        CALIBRATION_SEGMENT,
})


wandb.finish()

