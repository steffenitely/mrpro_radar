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
import os

# ══════════════════════════════════════════════════════════════════════════════
# ◀ CONFIGURE — File paths
# ══════════════════════════════════════════════════════════════════════════════
# Calibrate and rnd Movement
#Both done in one measurement
RADAR = "Data/RadarTest/radar_20260728_150309.npz"
  

RADAR2 = "Data/RadarTest/radar_20260729_145247.npz"

def load_radar(path: str):
    data = np.load(path, allow_pickle=True) #.npy for before 9.6
    radar_cube = data["radar_cube"] #old data before 9.6. was without cube so these steps need to be excluded
    timestamps = data["timestamps"]
    time_cube = data["time_cube"]
    return radar_cube, timestamps, time_cube


loaded_radar, radar_timestamps, radar_time_cube = load_radar(RADAR)

print(f"shape of radar cube: {loaded_radar.shape}")
print(f"shape of radar timestamps: {radar_timestamps.shape}")
print(f"shape of radar time cube: {radar_time_cube.shape}")

loaded_radar2, radar_timestamps2, radar_time_cube2 = load_radar(RADAR2)

print(f"shape of radar cube 2: {loaded_radar2.shape}")
print(f"shape of radar timestamps 2: {radar_timestamps2.shape}")
print(f"shape of radar time cube 2: {radar_time_cube2.shape}")