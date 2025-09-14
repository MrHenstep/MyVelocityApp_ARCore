import matplotlib.pyplot as plt

import numpy as np

import L2_lib_batch_fit_regression as bfr
import L1_lib_extraction_and_visualisation as exv
import L3_intrinsics_extrinsics as cprj

import persistence_helpers as phelp

###########################################################################################################

def get_depth_from_regression(file_path, depth_map_file_name_replacement,  row, regression_results, x, y, width):
    def get_depth_from_regression(file_path, depth_map_file_name_replacement, row, regression_results, x, y, width):
        """
        Computes the metric depth at a specific (x, y) coordinate using regression results and a depth map.
        Args:
            file_path (str): Path to the directory containing the depth map file.
            depth_map_file_name_replacement (str): String to replace "grey" in the depth map file name, if provided.
            row (list or pd.Series): Row containing metadata, including the depth map file name at index 3 and depth points file at index 0.
            regression_results (pd.DataFrame): DataFrame containing regression coefficients ('coef') and intercepts ('intercept') for each depth points file.
            x (int): X-coordinate in the depth map.
            y (int): Y-coordinate in the depth map.
            width (int): Width of the depth map (used to compute the linear index).
        Returns:
            float: The computed metric depth at the specified (x, y) location.
        """

    # Get rel depth from depth map
    depth_map_file = row[3]
    if depth_map_file_name_replacement:
        depth_map_file = depth_map_file.replace("grey", depth_map_file_name_replacement)

    depth_map = exv.read_float_data_as_nx6(file_path, depth_map_file)

    index = y * width + x

    rec_rel_depth = depth_map[index, 3] if index < depth_map.size else 0

    # # Get regression coefficients
    a = regression_results.loc[regression_results['depth_points_file'] == row[0], 'coef'].values[0]
    b = regression_results.loc[regression_results['depth_points_file'] == row[0], 'intercept'].values[0]

    # # Compute metric depth
    metric_depth = a / (rec_rel_depth + 1e-6 - b)

    return metric_depth

def get_projected_3d_coords(image_dimensions, principal_point, focal_length, width, height, x, y, metric_depth):
    """
    Projects 2D image coordinates into 3D camera coordinates using intrinsic camera parameters.
    Args:
        image_dimensions (tuple): The dimensions of the image (width, height).
        principal_point (tuple): The principal point of the camera (cx, cy) in pixel coordinates.
        focal_length (tuple): The focal length of the camera (fx, fy) in pixels.
        width (int or float): The reference width for scaling.
        height (int or float): The reference height for scaling.
        x (array-like): The x coordinates in the image.
        y (array-like): The y coordinates in the image.
        metric_depth (array-like or float): The depth value(s) in metric units.
    Returns:
        np.ndarray: The projected 3D camera coordinates in homogeneous form, shape (..., 4).
    """

    fx = focal_length[0] * image_dimensions[0] / width
    fy = focal_length[1] * image_dimensions[1] / height
    cx = principal_point[0] * image_dimensions[0] / width
    cy = principal_point[1] * image_dimensions[1] / height
    
    x_cam = (x - cx) / fx * metric_depth
    y_cam = -(y - cy) / fy * metric_depth
    z_cam = -metric_depth

    cam_homogeneous_points = np.array([x_cam, y_cam, z_cam, np.ones_like(x_cam)]).T

    return cam_homogeneous_points

def get_batch_3d_and_extrinsics(
        file_path, 
        matched_filename_table, 
        depth_map_file_name_replacement, 
        tracking_file_name_replacement, 
        width, 
        height, 
        frame_inclusion_list, 
        regression_results
    ):
    """
    Processes a batch of frames to extract 3D tracked points and camera extrinsic matrices.
    For each frame specified in `frame_inclusion_list`, this function:
    - Loads tracked 2D points from the tracking model file.
    - Computes metric depth for each point using regression results.
    - Reads camera intrinsics and extrinsics.
    - Projects 2D points to 3D camera coordinates.
    - Transforms 3D points to the ARCore reference frame using extrinsics.
    Args:
        file_path (str): Base directory containing data files.
        matched_filename_table (list): Table of matched filenames and metadata for each frame.
        depth_map_file_name_replacement (str): String to replace in depth map filenames.
        tracking_file_name_replacement (str): String to replace in tracking model filenames.
        width (int): Image width.
        height (int): Image height.
        frame_inclusion_list (list): Indices of frames to include in processing.
        regression_results (dict): Regression results for depth estimation.
    Returns:
        tuple:
            extrinsic_matrices (list): List of extrinsic matrices for each processed frame.
            tracked_points_cam (list): List of 3D tracked points in camera coordinates for each frame.
            tracked_points_ref (list): List of 3D tracked points in ARCore reference frame for each frame.
    """


    extrinsic_matrices = []
    tracked_points_cam = []
    tracked_points_ref = []

    for idx, row in enumerate(matched_filename_table):

        # skip if idx is not in depth_point_indices
        if idx not in frame_inclusion_list:
            continue

        
        # Read tracked points
        tracking_model_file = row[5]
        if tracking_file_name_replacement:
            tracking_model_file = tracking_model_file.replace("point", "point_" + tracking_file_name_replacement)

        tracked_points = exv.read_float_data_as_nxm(file_path, tracking_model_file)
        x = tracked_points[:, 0].astype(np.int32)
        y = tracked_points[:, 1].astype(np.int32)
    
        # get metric depth from regression
        metric_depth = get_depth_from_regression(file_path, depth_map_file_name_replacement, row, regression_results, x, y, width)

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

def print_transformation_data(depth_model_name, tracking_model_name, extrinsic_matrices, tracked_points_cam, tracked_points_ref):
    """
    Prints transformation and tracking data for a given depth and tracking model.
    Args:
        depth_model_name (str): Name of the depth model used.
        tracking_model_name (str): Name of the tracking model used.
        extrinsic_matrices (list of list of list of float): List of extrinsic matrices, each represented as a 2D list.
        tracked_points_cam (list of list of list of float): List of tracked points in the camera frame for each frame.
        tracked_points_ref (list of list of list of float): List of tracked points in the reference frame for each frame.
    Prints:
        - The names of the depth and tracking models.
        - The extrinsic matrices, row by row.
        - The tracked points in the camera frame, grouped by frame.
        - The tracked points in the reference frame, grouped by frame.
    """

    print(f"\n{depth_model_name}, {tracking_model_name}\n")

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

def write_transformation_data_to_file(filename, depth_model_name, tracking_model_name, extrinsic_matrices, tracked_points_cam, tracked_points_ref):
    """
    Writes transformation and tracking data to a specified file.
    Parameters:
        filename (str): Path to the output file where data will be written.
        depth_model_name (str): Name of the depth model used.
        tracking_model_name (str): Name of the tracking model used.
        extrinsic_matrices (list of list of list of float): List of extrinsic transformation matrices.
        tracked_points_cam (list of list of list of float): Tracked points in the camera frame, organized by frame.
        tracked_points_ref (list of list of list of float): Tracked points in the reference frame, organized by frame.
    The output file will contain:
        - The names of the depth and tracking models.
        - The extrinsic matrices.
        - The tracked points in both camera and reference frames, grouped by frame.
    """

    with open(filename, 'w') as f:
        f.write(f"\n{depth_model_name}, {tracking_model_name}\n\n")
    
        f.write("Extrinsic Matrices:\n")
        for matrix in extrinsic_matrices:
            for row in matrix:
                f.write(",".join(map(str, row)) + "\n")
            f.write("\n")
    
        f.write("Tracked points in camera frame:\n")
        for index, points in enumerate(tracked_points_cam):
            for point in points:
                f.write(f"Frame {index}," + ",".join(map(str, point)) + "\n")
    
        f.write("\nTracked points in reference frame:\n")
        for index, points in enumerate(tracked_points_ref):
            for point in points:
                f.write(f"Frame {index}," + ",".join(map(str, point)) + "\n")

def read_transformation_data_from_file(filename):
    """
    Reads transformation and tracking data from a formatted text file.
    The file is expected to contain:
        - A header line: "depth_model_name, tracking_model_name"
        - Sections:
            - "Extrinsic Matrices:" followed by one or more matrices (rows of comma-separated floats, matrices separated by blank lines)
            - "Tracked points in camera frame:" with lines of the format "Frame {index},<v1>,<v2>,<v3>,<v4?>"
            - "Tracked points in reference frame:" with lines of the format "Frame {index},<v1>,<v2>,<v3>,<v4?>"
    Args:
        filename (str): Path to the input file.
    Returns:
        tuple:
            depth_model_name (str): Name of the depth model from the header.
            tracking_model_name (str): Name of the tracking model from the header.
            extrinsic_matrices (list): List of extrinsic matrices (each a list of rows).
            tracked_points_cam (list): List of lists of tracked points in camera frame, indexed by frame.
            tracked_points_ref (list): List of lists of tracked points in reference frame, indexed by frame.
    Raises:
        ValueError: If the file is empty, missing required sections, or contains malformed lines.
    """

    with open(filename, "r") as f:
        # strip newlines; keep empty lines to detect matrix breaks
        lines = [line.strip() for line in f.readlines()]

    # --- Find header (skip any leading blank lines/BOM) ---
    i = 0
    while i < len(lines) and lines[i] == "":
        i += 1
    if i >= len(lines):
        raise ValueError("File is empty or contains only blank lines.")

    header_line = lines[i].lstrip("\ufeff")  # strip possible UTF-8 BOM
    # Expect "depth_model_name, tracking_model_name"
    if "," not in header_line:
        raise ValueError(f"Header line does not contain a comma: {header_line!r}")
    depth_model_name, tracking_model_name = [s.strip() for s in header_line.split(",", 1)]
    i += 1  # move past header

    # --- Prepare containers ---
    extrinsic_matrices = []
    tracked_points_cam = []
    tracked_points_ref = []

    section = None
    matrix_buf = []

    # --- Helper to flush any pending matrix rows ---
    def flush_matrix():
        nonlocal matrix_buf
        if matrix_buf:
            extrinsic_matrices.append(matrix_buf)
            matrix_buf = []

    # --- Main parse loop ---
    while i < len(lines):
        line = lines[i]
        i += 1

        # Section headers
        if line.startswith("Extrinsic Matrices:"):
            flush_matrix()  # just in case
            section = "extrinsics"
            continue
        if line.startswith("Tracked points in camera frame:"):
            flush_matrix()
            section = "cam_points"
            continue
        if line.startswith("Tracked points in reference frame:"):
            flush_matrix()
            section = "ref_points"
            continue

        # Blank line handling
        if line == "":
            # inside extrinsics, blank line separates individual matrices
            if section == "extrinsics":
                flush_matrix()
            # otherwise blanks are just separators; ignore
            continue

        # Content per section
        if section == "extrinsics":
            # Comma-separated floats form a row of the current matrix
            row_vals = [float(x) for x in line.split(",")]
            matrix_buf.append(row_vals)

        elif section in ("cam_points", "ref_points"):
            # Format written: "Frame {index},<v1>,<v2>,<v3>,<v4?>"
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0].startswith("Frame "):
                raise ValueError(f"Bad point line (missing 'Frame i,'): {line!r}")
            frame_idx = int(parts[0][len("Frame "):])
            coords = tuple(float(x) for x in parts[1:])
            target = tracked_points_cam if section == "cam_points" else tracked_points_ref
            # ensure list is large enough
            while len(target) <= frame_idx:
                target.append([])
            target[frame_idx].append(coords)

        else:
            # Lines before any section headers are expected to be blank; tolerate them
            continue

    # Flush any trailing matrix
    flush_matrix()

    return depth_model_name, tracking_model_name, extrinsic_matrices, tracked_points_cam, tracked_points_ref



###########################################################################################################

if __name__ == "__main__":

    ##########################################################################################################
    
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # DATA - A
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
    # # DATA - B
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
    # DATA - C
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

    ###########################################################################################################

    CONFIDENCE_LEVEL = 0.75
    DEPTH_MAX = 25.0
    DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, DEPTH_MAX)

    FRAME_INCLUSION_LIST = range(0, 100, 1)

    MATCH_TIMESTAMPS = False

    # Apply a sigmoid weight dropping towards zero above 7, width 1
    X_CUT = 7.0
    X_WIDTH = 1.0
    WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

    # INVERT_AXES = False
    # DISPLAY_PLOTS = False

    BATCH_NUMBER_LIST = [0,1,2,3]
    # BATCH_NUMBER_LIST = [0, 1]

    for batch_number in BATCH_NUMBER_LIST:

        TIMESTAMPS_TABLE = exv.read_timestamp_files(FILE_PATH, batch_number)

        if MATCH_TIMESTAMPS:
            MATCHED_INDICES = exv.find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
        else:
            MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)


        WIDTH = 640
        HEIGHT = 480

        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

        ###########################################################################################################



        # model_name_list = ["Phone_midas_v21_small", "midas_v21" , "dpt_large", "dpt_beit_large_512", "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"]
        depth_model_name_list = ["Phone_midas_v21_small", "depth_anything_v2_large"]
        tracking_model_name_list = ["Phone_openCV_LK", "CT2"]

        regression_results_dict = phelp.load_dataframe_dict_from_csv(FILE_PATH + "\\" + f"batch_{batch_number}_regression_results_dict.csv")

        for depth_model_name in depth_model_name_list:

            if "Phone" in depth_model_name:
                depth_map_file_name_replacement = None
            else:
                depth_map_file_name_replacement = "MOD_"+depth_model_name

            regression = regression_results_dict[depth_model_name]

            for tracking_model_name in tracking_model_name_list:

                if "Phone" in tracking_model_name:
                    tracking_map_file_name_replacement = None
                else:
                    tracking_map_file_name_replacement = "MOD_"+tracking_model_name

                depth_model_name, tracking_model_name, extrinsic_matrices, tracked_points_cam, tracked_points_ref = read_transformation_data_from_file(
                    FILE_PATH + "\\" + f"batch_{batch_number}_trajectories_3D_{depth_model_name}_{tracking_model_name}.csv"
                )

                print_transformation_data(
                    depth_model_name=depth_model_name, 
                    tracking_model_name=tracking_model_name, 
                    extrinsic_matrices=extrinsic_matrices, 
                    tracked_points_cam=tracked_points_cam, 
                    tracked_points_ref=tracked_points_ref
                )
