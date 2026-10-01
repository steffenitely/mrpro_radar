import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("Data/RadarTest/radar_data_20260528_145850.npy")

rx1 = data[0]
rx2 = data[1]

print("Data shape:", data.shape)

# ------------------------------------------------------------
# Estimate static background
# ------------------------------------------------------------

# First 50 scans should contain no movement
n_reference = 50

reference_rx1 = np.mean(rx1[:n_reference], axis=0)
reference_rx2 = np.mean(rx2[:n_reference], axis=0)

# ------------------------------------------------------------
# Subtract static background
# ------------------------------------------------------------

rx1_change = rx1 - reference_rx1
rx2_change = rx2 - reference_rx2

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 9),
    constrained_layout=True
)

# RX1 original
im0 = axes[0, 0].imshow(
    rx1,
    aspect="auto",
    interpolation="nearest"
)
axes[0, 0].set_title("RX1 — Original")
axes[0, 0].set_xlabel("Correlation bin")
axes[0, 0].set_ylabel("Scan")
fig.colorbar(im0, ax=axes[0, 0])


# RX1 change
limit = np.percentile(np.abs(rx1_change), 99)

im1 = axes[0, 1].imshow(
    rx1_change,
    #cmap="seismic",
    vmin=-limit,
    vmax=limit,
    aspect="auto",
    interpolation="nearest"
)
axes[0, 1].set_title("RX1 — Change from Static Background")
axes[0, 1].set_xlabel("Correlation bin")
axes[0, 1].set_ylabel("Scan")
fig.colorbar(im1, ax=axes[0, 1])


# RX2 original
im2 = axes[1, 0].imshow(
    rx2,
    aspect="auto",
    interpolation="nearest"
)
axes[1, 0].set_title("RX2 — Original")
axes[1, 0].set_xlabel("Correlation bin")
axes[1, 0].set_ylabel("Scan")
fig.colorbar(im2, ax=axes[1, 0])


# RX2 change
limit = np.percentile(np.abs(rx2_change), 99)

im3 = axes[1, 1].imshow(
    rx2_change,
    #cmap="seismic",
    vmin=-limit,
    vmax=limit,
    aspect="auto",
    interpolation="nearest"
)
axes[1, 1].set_title("RX2 — Change from Static Background")
axes[1, 1].set_xlabel("Correlation bin")
axes[1, 1].set_ylabel("Scan")
fig.colorbar(im3, ax=axes[1, 1])


plt.savefig(
    "Data/Backyard/radar_change.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()