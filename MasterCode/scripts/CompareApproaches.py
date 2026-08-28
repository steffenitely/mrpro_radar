import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import detrend
from sklearn.metrics import mean_squared_error

from mrpro.operators.PCACompressionOp import PCACompressionOp


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RADAR = "Data/RadarTest/radar_20260729_145247.npz"
MOTOR = "Data/TimeLogs/time_log_20260729_145342.json"
save_dir = "Data/calibration_plots/ApproachComparisonNew/00"

motor_pca_dir = os.path.join(save_dir, "Motor_vs_PCA")
regression_dir = os.path.join(save_dir, "RegressionModels")
single_reg_dir = os.path.join(save_dir, "RegressionModel_per_segment")
cal_test_dir = os.path.join(save_dir, "Calibration_vs_Test")
cal_matrix_dir = os.path.join(save_dir, "Calibration_Matrix")

for d in [motor_pca_dir, regression_dir, single_reg_dir, cal_test_dir, cal_matrix_dir]:
    os.makedirs(d, exist_ok=True)

SEGMENT_LENGTH = 40.0
CALIBRATION_SEGMENT = 3
TEST_SEGMENTS = range(7, 14)
N_COMPONENTS = 510
MOTOR_SCALE = 2000.0
MOTOR_EPOCH: datetime.datetime | None = None

# -----------------------------------------------------------------------------
# Input and preprocessing
# -----------------------------------------------------------------------------
def load_radar(path: str):
    data = np.load(path, allow_pickle=True)
    return data["radar_cube"], data["timestamps"], data["time_cube"]


def load_motor_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as file:
        entries = json.load(file)["motor_log"]
    return (
        np.array([entry["t"] for entry in entries], dtype=float),
        np.array([entry["position"] for entry in entries], dtype=float),
    )


def radar_times_to_seconds(radar_datetimes) -> np.ndarray:
    start = radar_datetimes[0]
    return np.array([(timestamp - start).total_seconds() for timestamp in radar_datetimes])


def align_motor_to_radar(
    motor_times: np.ndarray,
    radar_datetimes,
    motor_epoch: datetime.datetime | None,
    ) -> np.ndarray:
    offset = 0.0 if motor_epoch is None else (
        motor_epoch - radar_datetimes[0]
    ).total_seconds()
    return motor_times + offset

def remove_clutter(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=0, keepdims=True)


def basic_info_extraction2(radar_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rx1 = radar_cube[0, :, :]
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

# -----------------------------------------------------------------------------
#Align amd split
# -----------------------------------------------------------------------------
def align_and_split(
    radar_cube: np.ndarray,
    radar_times: np.ndarray,
    motor_times: np.ndarray,
    motor_positions: np.ndarray,
    segment_length: float,
) -> list[dict]:
    n_segments = int(np.floor(radar_times[-1] / segment_length))
    segments = []

    for segment_id in range(n_segments):
        start = segment_id * segment_length
        end = (segment_id + 1) * segment_length
        radar_mask = (radar_times >= start) & (radar_times < end)
        if np.count_nonzero(radar_mask) < 2:
            continue

        radar_segment = radar_cube[:, radar_mask, :]
        segment_radar_times = radar_times[radar_mask]
        motor_mask = (motor_times >= start) & (motor_times < end)

        # segments.append({
        #     "radar_segment": radar_segment,
        #     "radar_times": segment_radar_times,
        #     "motor_times": motor_times[motor_mask],
        #     "motor_positions": motor_positions[motor_mask],
        # })

        segments.append(
                (
                    radar_segment,
                    segment_radar_times,
                    motor_times[motor_mask],
                    motor_positions[motor_mask],
                ))

    return segments


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

class MotorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, values):
        return self.linear(values)

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


        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            print(f"Final loss: {loss.item():.6f}")
            break

        prev_loss = loss.item()

    model.eval()

    return model



def main() -> None:
    radar_cube, _, time_cube = load_radar(RADAR)
    raw_motor_times, motor_positions = load_motor_log(MOTOR)
    radar_times = radar_times_to_seconds(time_cube[0, :])
    motor_times = align_motor_to_radar(raw_motor_times, time_cube[0, :], MOTOR_EPOCH)

    rx1 = radar_cube[0, :, :]
    rx2 = radar_cube[1, :, :] 

    rx1 = remove_clutter(rx1)
    rx2 = remove_clutter(rx2)
    rx1 = detrend(rx1)
    rx2 = detrend(rx2)

    radar_cube[0, :, :] = rx1
    radar_cube[1, :, :] = rx2




    segments = align_and_split(radar_cube, radar_times, motor_times, motor_positions, SEGMENT_LENGTH)

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

        rx1, rx2 = basic_info_extraction2(radar_cube_seg)
        rx1_cf, rx2_cf = rx1, rx2


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
        # ─────────────────────────────────────
        # Train model
        # ─────────────────────────────────────

        

        model = train_model(X_filtered, y_filtered, i)
        
        model_path = os.path.join(save_dir, f"model_segment_{i}.pkl")


        torch.save(model.state_dict(), model_path)
        

        slope = model.linear.weight.item()
        intercept = model.linear.bias.item()

        
        # Log training metrics
        

        print(f"slope: {slope:.4f}, intercept: {intercept:.4f}")

        # store results
        calibration_models.append({
            "segment": i,
            "model": model,
            "slope": slope,
            "intercept": intercept,
            "mean": mean, 
            "pca_operator": op  
        })

        
        # ─────────────────────────────────────
        # plot per segment
        # ─────────────────────────────────────
        #x_line = np.linspace(X.min(), X.max(), 100)
        x_line = np.linspace(-0.065, 0.065, 500)  # fixed range for all segments
        x_t = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            y_line = model(x_t).squeeze().numpy()

        plt.figure()
        plt.scatter(X_filtered, y_filtered, s=2, alpha=0.4)
        plt.plot(x_line, y_line, color='orange')
        plt.ylim(0, 3)
        plt.xlabel("PCA")
        plt.ylabel("Position in cm")
        plt.title(f"Segment {i}: Slope = {slope:.3f}; Intercept = {intercept:.3f} ")
        
        filename = os.path.join(single_reg_dir, f"RegressionModel_segment{i}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    
        plt.close()
        
        
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
        op = calibration["pca_operator"]
        model = calibration["model"]


        for j, test in enumerate(all_segments):

            test_id = test["segment"]
            print(f"Calibration {i} -> Test {j}")


            # -------------------------------
            # Extract radar features
            # -------------------------------
            rx1, rx2 = basic_info_extraction2(test["radar_cube"])
            rx1_cf, rx2_cf = rx1, rx2

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

            error_rmse = rmse(y_motor, y_interp)
            error_mae = mae(y_motor, y_interp)
            error_max = max_abs_error(y_motor, y_interp)

            

            model_path = os.path.join(save_dir, f"model_segment_{i}.pt")
            torch.save(model.state_dict(), model_path)
            

            # -------------------------------
            # RMSE and Plots
            # -------------------------------
            
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

            filename = os.path.join(cal_test_dir, f"cal_{i}_test_{j}.png")
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
    plt.colorbar(label="RMSE in cm")
    plt.xlabel("Test segment")
    plt.ylabel("Calibration segment")
    plt.title("Calibration Transfer Matrix - RMSE")

    filename = os.path.join(cal_matrix_dir, "calibration_matrix_rmse.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.close()

    # ─────────────────────────────────────
    # Plot and save MAE calibration matrix
    # ─────────────────────────────────────
    plt.figure(figsize=(7,6))

    plt.imshow(
        calibration_matrix_mae[:6, :],
        aspect="auto"
    )
    plt.colorbar(label="MAE in cm")
    plt.xlabel("Test segment")
    plt.ylabel("Calibration segment")
    plt.title("Calibration Transfer Matrix - MAE")

    filename = os.path.join(cal_matrix_dir, "calibration_matrix_mae.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.close()

    # ─────────────────────────────────────
    # Plot and save Max Error calibration matrix
    # ─────────────────────────────────────
    plt.figure(figsize=(7,6))

    plt.imshow(
        calibration_matrix_max_error[:6, :],
        aspect="auto"
    )
    plt.colorbar(label="Max Abs Error in cm")
    plt.xlabel("Test segment")
    plt.ylabel("Calibration segment")
    plt.title("Calibration Transfer Matrix - Max Absolute Error")

    filename = os.path.join(cal_matrix_dir, "calibration_matrix_max_error.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.close()

    print("mean RMSE:", np.mean(calibration_matrix_rmse[:6, :]))
    print("mean MAE:", np.mean(calibration_matrix_mae[:6, :]))
    print("mean Max Error:", np.mean(calibration_matrix_max_error[:6, :]))







if __name__ == "__main__":
    main()