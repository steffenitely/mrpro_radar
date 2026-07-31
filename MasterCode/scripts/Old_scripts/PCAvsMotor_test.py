"""
plot_pca_vs_motor.py
────────────────────
Align motor log (relative time) with radar PCA data (absolute datetimes)
and plot both on the same time axis starting at t = 0.

CONFIGURE the three sections marked with  ◀ CONFIGURE  before running.
"""

import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, detrend

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE 1 — Action → physical position mapping
#   Add / rename keys to match every label that appears in your motor_log.
#   Values are the numeric position you want plotted (e.g. mm, degrees, …).
# ══════════════════════════════════════════════════════════════════════════════
ACTION_MAP: dict[str, float] = {
    "0": 0.0,    # e.g. home / reference position
    "A": 2000.0,    # e.g. first target position
    # "B": 2.0,
    # "C": 3.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE 2 — File paths
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_LOG_PATH = "TimeLogs/time_log_20260611_120330.json"   # path to your JSON file
RADAR_PATH = "RadarTest/radar_20260611_120326.npz"
# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE 3 — Time alignment
#
#   The motor log uses relative seconds (t ≈ 0 … N).
#   The radar uses absolute datetime objects.
#
#   You need to express "when did t=0 of the motor log happen, in radar time?"
#   Two common cases:
#
#   CASE A — You know the absolute wall-clock time of motor t=0:
#       Set MOTOR_EPOCH to that datetime, e.g.:
#           MOTOR_EPOCH = datetime.datetime(2026, 6, 9, 17, 36, 10, 469572)
#
#   CASE B — Both recordings simply started at the same moment
#       (i.e. motor t=0 == radar frame 0):
#           MOTOR_EPOCH = None   ← the script aligns their t=0 automatically
#
# ══════════════════════════════════════════════════════════════════════════════
MOTOR_EPOCH: datetime.datetime | None = None


# ─────────────────────────────────────────────────────────────────────────────
# 1.1. Load & parse the motor log
# ─────────────────────────────────────────────────────────────────────────────
def load_motor_log(path: str) -> tuple[np.ndarray, np.ndarray]:
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

# ─────────────────────────────────────────────────────────────────────────────
# 1.2. Load Radar data
# ─────────────────────────────────────────────────────────────────────────────
def load_radar(path: str):
    data = np.load(path, allow_pickle=True) #.npy for before 9.6
    radar_cube = data["radar_cube"] #old data before 9.6. was without cube so these steps need to be excluded
    timestamps = data["timestamps"]
    time_cube = data["time_cube"]
    return radar_cube, timestamps, time_cube



# ─────────────────────────────────────────────────────────────────────────────
# 2. Convert radar datetimes → seconds relative to radar t=0
# ─────────────────────────────────────────────────────────────────────────────
def radar_times_to_sec(radar_datetimes: list[datetime.datetime]) -> np.ndarray:
    """Subtract the first datetime so radar starts at 0."""
    t0 = radar_datetimes[0]
    return np.array([(dt - t0).total_seconds() for dt in radar_datetimes])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shift motor times so they share the same t=0 as the radar
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
# 4. PCA Extraction
# ─────────────────────────────────────────────────────────────────────────────
def remove_clutter(radar_data_data):
    # radar_data: (T, R)
    return radar_data_data - np.mean(radar_data_data, axis=0, keepdims=True)

def extract_PCA(radar_cube):
    rx1 = radar_cube[0, :, :] #old was without time stamps so directly data
    rx2 = radar_cube[1, :, :]

    rx1_cf = remove_clutter(rx1)
    rx2_cf = remove_clutter(rx2)

    # detrending?
    rx1_cf = detrend(rx1_cf)
    rx2_cf = detrend(rx2_cf)

    X_both = np.hstack([rx1_cf, rx2_cf])  # (timeframes, 2*510)
    np.shape(X_both)
    # Mean-center each feature column (required for PCA)
    X_both = X_both - X_both.mean(axis=0)
    print("Feature matrix shape:", X_both.shape)  # if both antennas(time:, 2N)

    pca = PCA(n_components=510)
    comps = pca.fit_transform(X_both)  # 
    return comps
# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot(
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
# 5. Main — wire everything together
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load motor log ────────────────────────────────────────────────────────
    motor_times, motor_positions = load_motor_log(MOTOR_LOG_PATH)

    # ── ◀ INSERT your radar data here ─────────────────────────────────────────
    # Replace the two lines below with your actual loading code.
    radar_cube, timeframes, timecube = load_radar(RADAR_PATH)
    comps = extract_PCA(radar_cube)
    radar_datetimes = timecube[0,:]
    pca_values = comps[:,:4]


    #
    # radar_datetimes : list[datetime.datetime]   — one per PCA frame
    # pca_values      : np.ndarray, shape (len(radar_datetimes),)
    #
    # Example (using time_cube):
    #
    #   # time_cube[0] = frame start times as datetimes, time_cube[1] = end times
    #   radar_datetimes = list(time_cube[0])          # or time_cube[1]
    #   pca_values      = pca_result[:, 0]            # first principal component
    #


    # ── Convert radar times → seconds from t=0 ───────────────────────────────
    radar_t_sec = radar_times_to_sec(radar_datetimes)

    # ── Shift motor times to the same reference ───────────────────────────────
    motor_t_aligned = align_motor_to_radar(
        motor_times, radar_datetimes, MOTOR_EPOCH
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot(
        radar_times_sec=radar_t_sec,
        pca_values=pca_values,
        motor_times_aligned=motor_t_aligned,
        motor_positions=motor_positions,
        pca_label="PC 1",
        position_unit="(label)",
    )


if __name__ == "__main__":
    main()