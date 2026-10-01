import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

# Load data
#data_path = Path(__file__).resolve().parents[2] / "Data" / "RadarTest" / "radar_20260618_153139.npy"
data = np.load("Data/RadarTest/OLD/radar_data_20260603_144110.npy")  # shape: (2, 441, 510)

#print(data.shape)

rx1, rx2 = data[0], data[1]

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


row_idx = rx1.shape[0] // 2  # middle row
rx1_peak = np.argmax(rx1[row_idx])
rx2_peak = np.argmax(rx2[row_idx])

time_difference = abs(rx1_peak-rx2_peak)*112e-12 # in ps
distance = time_difference*c
print(f"The distance is {distance} meters")

fig2, ax = plt.subplots(figsize=(10, 4))

ax.plot(rx1[row_idx], label="Channel 0")
ax.plot(rx2[row_idx], label="Channel 1", linestyle="--")

ax.set_title(f"Correlation vs Samples (Row {row_idx})")
ax.set_xlabel("Sample index (0–509)")
ax.set_ylabel("Correlation strength")

ax.legend()
ax.grid(True)

plt.show()