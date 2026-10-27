#load in motor movement data and visualize it


import matplotlib.pyplot as plt
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import calibrate_pca_centering_options as shared

MOTOR_PATH1 = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_133235.json")
MOTOR_PATH2 = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_131300.json")
MOTOR_PATH3 = Path("Data/Input/CalibrationTrial/MotorLogs/time_log_20261022_142932.json")


visualization_output_dir = Path("Data/Output/CalibrationTrial/Visualizations/CalibrationMovements")

