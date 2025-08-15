import lib_extraction_and_visualisation as exv
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

def batch_fit_regression_model(file_path, batch_number, depth_point_indices, confidence_level=0.7, depth_range_for_colour_map = (0, 25), weights_sigmoid=None, display_plots=True, match_timestamps=False):

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

        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])
        if (depth_points_index not in depth_point_indices):
            continue

        # combined data points: x, y, depth, confidence, depth map value
        combined_data_points = exv.get_depth_point_vs_map_data(file_path, depth_points_file, depth_map_file, confidence_level)
        # exv.plot_histograms_and_regression(combined_data_points, depth_range_for_colour_map)

        model, params = fit_regression_model(combined_data_points, weights_sigmoid=weights_sigmoid)

        line_points = generate_prediction_points(model, depth_range_for_colour_map)

        # plot histograms and regression
        if display_plots:
            exv.plot_histograms_and_regression(combined_data_points, depth_range_for_colour_map, line_points)

        print(file_name_row)

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

def plot_regression_fits(df_results, invert_axes, depth_max):

    # Plot all lines on the same graph
    plt.figure()
    for index, row in df_results.iterrows():
        a = row['coef']
        b = row['intercept']

        eps = 1e-3
        if invert_axes:
            x_line = np.linspace(b+eps, 1.0+eps, 100).reshape(-1, 1)
            print(index, a / (0.2 -b))
            y_line = a / (x_line - b)
        else:
            x_line = np.linspace(0.0+eps, depth_max+eps, 100).reshape(-1, 1)
            print(index, a / 10 + b)
            y_line = a / x_line + b


        plt.plot(x_line, y_line, label=f'Row {index}')

    if invert_axes:
        plt.xlabel('metric_depth_line')
        plt.ylabel('rec_rel_depth_line')
        plt.ylim(0, depth_max)

    else:
        plt.xlabel('rec_rel_depth_line')
        plt.ylabel('metric_depth_line')
        plt.ylim(0, 1)

        
    plt.grid(True)
    plt.title('Metric Depth vs Reciprocal Relative Depth')
    # plt.legend()
    plt.show()

##################################################################################################################


if (__name__ == "__main__"):

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported\\20250812_1_(frametiming)(indoors)(motion)"
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported\\20250812_2_(frametiming)(outdoors)(motion)"
    FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported\\20250813_1_(5fps)(outside)"
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported"

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

    ###########################################################################################################

    df_results = batch_fit_regression_model(FILE_PATH, BATCH_NUMBER, DEPTH_POINTS_INDICES, CONFIDENCE_LEVEL, DEPTH_RANGE_FOR_COLOUR_MAP, weights_sigmoid=WEIGHTS_SIGMOID, display_plots=False, match_timestamps=MATCH_TIMESTAMPS)

    plot_regression_fits(df_results, INVERT_AXES, DEPTH_MAX)

