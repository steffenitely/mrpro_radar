import numpy as np
import matplotlib.pyplot as plt

# Load data
#data_path = Path(__file__).resolve().parents[2] / "Data" / "RadarTest" / "radar_20260618_153139.npy"
data = np.load("Data/RadarTest/OLD/radar_data_20260603_144110.npy")  # shape: (2, 441, 510)

#print(data.shape)

rx1, rx2 = data[0], data[1]

# Shared color scale (important for correlation comparability)
vmin = min(rx1.min(), rx2.min())
vmax = max(rx1.max(), rx2.max())

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