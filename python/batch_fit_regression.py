import lib_extraction_and_visualisation as exv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib_batch_fit_regression as bfr


##########################################################################################################

# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_1_(frametiming)(indoors)(motion)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_2_(frametiming)(outdoors)(motion)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250813_1_(5fps)(outside)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250818_1_(5fps)(outside)(tracking)"
FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_1_(5fps)(car)(Ziggy)"

###########################################################################################################


BATCH_NUMBER = 1
CONFIDENCE_LEVEL = 0.75
DEPTH_MAX = 25.0
DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, DEPTH_MAX)

DEPTH_POINTS_INDICES = range(0, 100, 1)

MATCH_TIMESTAMPS = False

# Apply a sigmoid weight dropping towards zero above 7, width 1
X_CUT = 7.0
X_WIDTH = 1.0
WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

INVERT_AXES = False
DISPLAY_PLOTS = True

###########################################################################################################

df_results = bfr.batch_fit_regression_model(FILE_PATH, BATCH_NUMBER, DEPTH_POINTS_INDICES, CONFIDENCE_LEVEL, DEPTH_RANGE_FOR_COLOUR_MAP, weights_sigmoid=WEIGHTS_SIGMOID, display_plots=DISPLAY_PLOTS, match_timestamps=MATCH_TIMESTAMPS)

bfr.plot_regression_fits(df_results, INVERT_AXES, DEPTH_MAX)

