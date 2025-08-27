import L1_lib_extraction_and_visualisation as exv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##########################################################################################################


def fit_regression_model(combined_data_points, weights_sigmoid = None):

    # extract cols 2 and 4 and then regress, using col 3 as weights
    data_x = combined_data_points[:, 2]
    data_y = combined_data_points[:, 4]
    data_w = combined_data_points[:, 3]

    # Exclude data points where data_y is NaN
    mask = ~np.isnan(data_y)
    data_x = data_x[mask]
    data_y = data_y[mask]
    data_w = data_w[mask]

    if weights_sigmoid:
        sigmoid = lambda x: 1 / (1 + np.exp((x - weights_sigmoid[0]) / weights_sigmoid[1]))
        data_w *= sigmoid(data_x)

    # transform x to reciprocal
    data_x_inv = 1 / data_x

    # perform regression
    model = LinearRegression()
    data_x_reshaped = data_x_inv.reshape(-1, 1)
    model.fit(data_x_reshaped, data_y, sample_weight=data_w)

    coef = model.coef_[0]
    intercept = model.intercept_
    r2_score = model.score(data_x_reshaped, data_y, sample_weight=data_w)

    return model, (coef, intercept, r2_score)

def generate_prediction_points(model, depth_range_for_colour_map, num_points=100):

    # generate prediction points
    eps = 1e-5
    line_x = np.linspace(depth_range_for_colour_map[0]+eps, depth_range_for_colour_map[1]+eps, num_points).reshape(-1, 1)
    line_x_inv = 1 / line_x
    line_y = model.predict(line_x_inv)
    line_points = np.column_stack((line_x, line_y))

    return line_points

def batch_fit_regression_model(
        file_path, 
        batch_number, 
        depth_point_indices, 
        depth_map_file_name_replacement = None, 
        confidence_level=0.7, 
        depth_range_for_colour_map = (0, 25), 
        weights_sigmoid=None, 
        display_plots=True, 
        match_timestamps=False):

    timestamps_table = exv.read_timestamp_files(file_path, batch_number)

    if match_timestamps:
        matched_indices = exv.find_closest_timestamp_matches(timestamps_table, 3, 2, direction='both')
    else:
        matched_indices = exv.get_all_indices(file_path, batch_number)

    matched_filename_table = exv.get_matched_filenames(matched_indices, file_path, batch_number)

    results = []

    for _, file_name_row in enumerate(matched_filename_table):

         # ROW = MATCHED_FILENAME_TABLE[0]
        depth_points_file = file_name_row[0]
        depth_map_file = file_name_row[3]

        if depth_map_file_name_replacement:
            depth_map_file = depth_map_file.replace("grey", depth_map_file_name_replacement)

        if "grey" in depth_map_file or "midas_v21" in depth_map_file:
            width_crop_size = (640 - 480) / 2
            height_crop_size = 0
        else:
            width_crop_size = 0
            height_crop_size = 0

        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])
        if (depth_points_index not in depth_point_indices):
            continue

        # combined data points: x, y, depth, confidence, depth map value

        combined_data_points = exv.get_depth_point_vs_map_data(file_path, depth_points_file, depth_map_file, confidence_level, width_crop_size=width_crop_size, height_crop_size=height_crop_size)
        # exv.plot_histograms_and_regression(combined_data_points, depth_range_for_colour_map)

        model, params = fit_regression_model(combined_data_points, weights_sigmoid=weights_sigmoid)

        line_points = generate_prediction_points(model, depth_range_for_colour_map)

        # plot histograms and regression
        if display_plots:
            exv.plot_histograms_and_regression(combined_data_points, depth_range_for_colour_map, line_points)

        # print(file_name_row)

        # Append results to the list
        results.append({
            "depth_points_file": depth_points_file,
            "depth_map_file": depth_map_file,
            "coef": params[0],
            "intercept": params[1],
            "r2_score": params[2]
        })
        
    df_results = pd.DataFrame(results)
    # print(df_results)

    return df_results

def plot_regression_fits(df_results, invert_axes, depth_max, model_name):

    # Plot all lines on the same graph
    plt.figure()
    for index, row in df_results.iterrows():
        a = row['coef']
        b = row['intercept']

        eps = 1e-3
        if invert_axes:
            x_line = np.linspace(b+eps, 1.0+eps, 100).reshape(-1, 1)
            # print(index, a / (0.2 -b))
            y_line = a / (x_line - b)
        else:
            x_line = np.linspace(0.0+eps, depth_max+eps, 100).reshape(-1, 1)
            # print(index, a / 10 + b)
            y_line = a / x_line + b


        plt.plot(x_line, y_line, label=f'Row {index}')

    if invert_axes:
        plt.ylabel('metric depth')
        plt.xlabel('rec rel depth')
        plt.ylim(0, depth_max)

    else:
        plt.ylabel('rec rel depth')
        plt.xlabel('metric depth')
        plt.ylim(0, 1)

        
    plt.grid(True)
    plt.title(model_name)
    # plt.legend()
    plt.show()

def calculate_regression_line_points(a, b, depth_max, invert_axes, num_points=100):
    eps = 1e-3
    if invert_axes:
        x_line = np.linspace(b + eps, 1.0 + eps, num_points).reshape(-1, 1)
        y_line = a / (x_line - b)
    else:
        x_line = np.linspace(0.0 + eps, depth_max + eps, num_points).reshape(-1, 1)
        y_line = a / x_line + b
    return x_line, y_line

def plot_multiple_regression_fits(
    list_of_df_results, 
    list_of_invert_axes, 
    list_of_depth_max, 
    list_of_model_names,
    ncols=4
):
    n = len(list_of_df_results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, (df_results, invert_axes, depth_max, model_name) in enumerate(zip(
        list_of_df_results, list_of_invert_axes, list_of_depth_max, list_of_model_names
    )):
        ax = axes[idx // ncols][idx % ncols]
        for index, row in df_results.iterrows():
            a = row['coef']
            b = row['intercept']
            x_line, y_line = calculate_regression_line_points(a, b, depth_max, invert_axes)
            ax.plot(x_line, y_line, label=f'Row {index}')
        if invert_axes:
            ax.set_ylabel('metric depth')
            ax.set_xlabel('rec rel depth')
            ax.set_ylim(0, depth_max)
        else:
            ax.set_ylabel('rec rel depth')
            ax.set_xlabel('metric depth')
            ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(model_name)
        # ax.legend()
    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])
    plt.tight_layout()
    plt.show()
##################################################################################################################


if (__name__ == "__main__"):

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"



    BATCH_NUMBER = 0
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
    DISPLAY_PLOTS = False

    ###########################################################################################################
    depth_map_file_name_replacement = None

    df_results = batch_fit_regression_model(
        FILE_PATH,
        BATCH_NUMBER,
        DEPTH_POINTS_INDICES,
        depth_map_file_name_replacement=depth_map_file_name_replacement,
        confidence_level=CONFIDENCE_LEVEL,
        depth_range_for_colour_map=DEPTH_RANGE_FOR_COLOUR_MAP,
        weights_sigmoid=WEIGHTS_SIGMOID,
        display_plots=DISPLAY_PLOTS,
        match_timestamps=MATCH_TIMESTAMPS
    )

    plot_regression_fits(df_results, INVERT_AXES, DEPTH_MAX)

