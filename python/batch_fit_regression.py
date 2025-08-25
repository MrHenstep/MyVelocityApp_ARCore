import sys
import lib_extraction_and_visualisation as exv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib_batch_fit_regression as bfr

import coordinate_projection_3d as cprj

##########################################################################################################

# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_2_(test-static)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_2_(5ps)(test)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_1_(5fps)(car)(Ziggy)"
FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_3_(test_rotation)"

###########################################################################################################

def get_depth_from_regression(file_path, row, regression_results, x, y, width):

    # Get rel depth from depth map
    depth_map = exv.read_float_data_as_nx6(file_path, row[3])

    index = y * width + x

    rec_rel_depth = depth_map[index, 3] if index < depth_map.size else 0

    # # Get regression coefficients
    a = regression_results.loc[regression_results['depth_points_file'] == row[0], 'coef'].values[0]
    b = regression_results.loc[regression_results['depth_points_file'] == row[0], 'intercept'].values[0]

    # # Compute metric depth
    metric_depth = a / (rec_rel_depth + 1e-6 - b)

    return metric_depth

def get_projected_3d_coords(image_dimensions, principal_point, focal_length, width, height, x, y, metric_depth):

    fx = focal_length[0] * image_dimensions[0] / width
    fy = focal_length[1] * image_dimensions[1] / height
    cx = principal_point[0] * image_dimensions[0] / width
    cy = principal_point[1] * image_dimensions[1] / height
    
    x_cam = (x - cx) / fx * metric_depth
    y_cam = -(y - cy) / fy * metric_depth
    z_cam = -metric_depth

    cam_homogeneous_points = np.array([x_cam, y_cam, z_cam, np.ones_like(x_cam)]).T

    return cam_homogeneous_points

def get_batch_3d_and_extrinsics(file_path, matched_filename_table, width, height, depth_point_indices, regression_results):

    extrinsic_matrices = []
    tracked_points_cam = []
    tracked_points_ref = []

    for idx, row in enumerate(matched_filename_table):

        # skip if idx is not in depth_point_indices
        if idx not in depth_point_indices:
            continue


        # Read tracked points
        tracked_points = exv.read_float_data_as_nx4(file_path, row[5])
        x = tracked_points[:, 0].astype(np.int32)
        y = tracked_points[:, 1].astype(np.int32)
    
        # get metric depth from regression
        metric_depth = get_depth_from_regression(file_path, row, regression_results, x, y, width)

        # camera intrinsics
        image_dimensions, principal_point, focal_length = cprj.read_intrinsics_bin(file_path + "\\" +row[8])

        # project to Camera 3D coords
        cam_homogeneous_points = get_projected_3d_coords(image_dimensions, principal_point, focal_length, width, height, x, y, metric_depth)    

        # extrinsics
        extrinsic_matrix = cprj.read_extrinsics_matrix(file_path + "\\" + row[6])
        extrinsic_matrices.append(extrinsic_matrix)

        # transform to ARCore reference frame
        ref_homogeneous_points = (extrinsic_matrix @ cam_homogeneous_points.T).T

        tracked_points_cam.append(cam_homogeneous_points)
        tracked_points_ref.append(ref_homogeneous_points)

    return extrinsic_matrices, tracked_points_cam, tracked_points_ref

def print_transformation_data (extrinsic_matrices, tracked_points_cam, tracked_points_ref):
    print("Extrinsic Matrices:")
    for matrix in extrinsic_matrices:
        for row in matrix:
            print(",".join(map(str, row)))
        print()  # New line for better readability

    print("Tracked points in camera frame:")
    for index, points in enumerate(tracked_points_cam):
        for point in points:
            print(f"Frame {index},", ",".join(map(str, point)))
        

    print("\nTracked points in reference frame:")
    for index, points in enumerate(tracked_points_ref):
        for point in points:
            print(f"Frame {index},", ",".join(map(str, point)))
        



###########################################################################################################

if __name__ == "__main__":

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
    DISPLAY_PLOTS = True

    TIMESTAMPS_TABLE = exv.read_timestamp_files(FILE_PATH, BATCH_NUMBER)

    if MATCH_TIMESTAMPS:
        MATCHED_INDICES = exv.find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
    else:
        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, BATCH_NUMBER)

    # Get matched filenames
    # Col 0 - depth points
    # Col 1 - camera image
    # Col 2 - depth map colour
    # Col 3 - depth map grey
    # Col 4 - confidence points
    # Col 5 - tracked points
    # Col 6 - extrinsic mx
    # Col 7 - texture intrinsics
    # Col 8 - camera intrinsics

    MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, BATCH_NUMBER)

    ###########################################################################################################

    REGRESSION = bfr.batch_fit_regression_model(FILE_PATH, BATCH_NUMBER, DEPTH_POINTS_INDICES, CONFIDENCE_LEVEL, DEPTH_RANGE_FOR_COLOUR_MAP, weights_sigmoid=WEIGHTS_SIGMOID, display_plots=DISPLAY_PLOTS, match_timestamps=MATCH_TIMESTAMPS)

    bfr.plot_regression_fits(REGRESSION, INVERT_AXES, DEPTH_MAX)

    ###########################################################################################################

    WIDTH = 640
    HEIGHT = 480

    extrinsic_matrices, tracked_points_cam, tracked_points_ref = get_batch_3d_and_extrinsics(FILE_PATH, MATCHED_FILENAME_TABLE, WIDTH, HEIGHT, DEPTH_POINTS_INDICES, REGRESSION)

    print_transformation_data(extrinsic_matrices, tracked_points_cam, tracked_points_ref)
