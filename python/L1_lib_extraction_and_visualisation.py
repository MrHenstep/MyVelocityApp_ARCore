import os
import re
import numpy as np
import struct
import matplotlib.pyplot as plt

########################################################################################################################
### extracting the time stamps and then matching depth images with camera images

def read_timestamp_files(directory, batch_number):
    """
    Reads binary timestamp files from a specified directory for a given batch number, extracts data, and returns it as a NumPy array.
    The function searches for files matching the pattern 'batch_{batch_number}_timestamps_{nn}.bin', where {nn} is an integer.
    Each file is expected to contain exactly 4 floats (16 bytes) in binary format.
    Parameters:
        directory (str): Path to the directory containing the timestamp files.
        batch_number (int): The batch number used to filter relevant files.
    Returns:
        np.ndarray: A NumPy array of shape (M, 5), where M is the number of files found.
                    The first column contains the file index (nn), and the next four columns contain the float values read from each file (scaled down by 1e9).
    Raises:
        ValueError: If any file does not contain exactly 4 floats (16 bytes).
    """
    # Pattern to match filenames like timestamps_1.bin, timestamps_23.bin, etc.
    pattern = re.compile(rf"batch_{batch_number}_timestamps_(\d+)\.bin")
    data = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            nn = int(match.group(1))
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                # Read 4 floats (4 bytes each => 16 bytes total)
                raw = f.read(4 * 4)
                if len(raw) != 16:
                    raise ValueError(f"File {filename} does not contain exactly 4 floats.")
                floats = struct.unpack('ffff', raw)
                data.append([nn] + list(floats))

    # Convert to NumPy array with shape (M, 5)
    data.sort(key=lambda x: x[0])
    data_array = np.array(data, dtype=np.float32)
    data_array[:,1:] /= 1e9
    # data_array[:, 1:] *= 3e1  # Scale the last three columns by 30
    return data_array

def find_closest_timestamp_matches(timestamps_table, search_col, target_col, direction='both'):
    """
    For each row, finds the index of the row in target_col whose value is closest
    to the value in the search_col of the current row.

    Parameters:
        array (np.ndarray): Input N x 5 array.
        search_col (int): Index of the column to take search values from.
        target_col (int): Index of the column to search in.
        direction (str): 'both', 'up', or 'down' for match direction.
        
    Returns:
        np.ndarray: Nx2 array of [original_index, matched_index] for each row.
    """
    search_values = timestamps_table[:, search_col]
    target_values = timestamps_table[:, target_col]

    match_indices = []

    for i, value in enumerate(search_values):
        if direction == 'both':
            diffs = np.abs(target_values - value)
        elif direction == 'up':
            diffs = np.where(target_values >= value, target_values - value, np.inf)
        elif direction == 'down':
            diffs = np.where(target_values <= value, value - target_values, np.inf)
        else:
            raise ValueError("direction must be 'both', 'up', or 'down'")

        closest_idx = np.argmin(diffs)
        match_indices.append([i, closest_idx])

    return np.array(match_indices, dtype=int)

def print_closest_ts_match(timestamps_array, match_indices):
   
    print("\nClosest Matches difference ...\n")
    for i, indices in enumerate(match_indices):

        print(f"Row {i}: diff, Frame ts {timestamps_array[indices[1], 1]-timestamps_array[indices[0], 3]:.6f}, Camera ts {timestamps_array[indices[1], 2]-timestamps_array[indices[0], 3]:.6f}, Depth ts {timestamps_array[indices[1], 3]-timestamps_array[indices[0], 3]:.6f}, Confidence ts {timestamps_array[indices[1], 4]-timestamps_array[indices[0], 3]:.6f}")



def get_all_indices(directory, batch_number):
    """
    Retrieves and returns all index values from filenames in a specified directory that match a given batch number pattern.

    The function searches for files named in the format 'batch_{batch_number}_depth_points_{index}.bin',
    extracts the index values, sorts them, and returns a NumPy array where each row contains the index value twice.

    Args:
        directory (str): Path to the directory containing the files.
        batch_number (int or str): The batch number to match in the filenames.

    Returns:
        numpy.ndarray: A 2D array of shape (N, 2), where N is the number of matching files, and each row contains [index, index].
    """
    pattern = re.compile(rf"batch_{batch_number}_depth_points_(\d+)\.bin")
    indices = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            indices.append(idx)
    indices.sort()
    match_indices = np.column_stack((indices, indices))
    return match_indices

def get_matched_filenames(match_indices, directory, batch_number):
    """
    Generates a table of filenames for matched indices, checking their existence in the specified directory.
    For each pair of indices in `match_indices`, constructs filenames for various data types related to a batch,
    checks if each file exists in the given `directory`, and returns either the filename or "n/a" if not found.
    Args:
        match_indices (list of tuple): List of index pairs (nn, mm) to match files.
        directory (str): Path to the directory containing the files.
        batch_number (int): Batch number used in the filenames.
    Returns:
        list of list: A table where each row corresponds to a pair of indices and contains the filenames (or "n/a")
                      for the following files:
                        - depth_points
                        - depth_map_camera
                        - depth_map_colour
                        - depth_map_grey
                        - confidence_points
                        - tracked_point
                        - extrinsic_matrix
                        - texture_intrinsics
                        - camera_intrinsics
    """


    filename_table = []
    # directory = "../exported"

    for idx_pair in match_indices:
        nn = idx_pair[0]
        mm = idx_pair[1]
        depth_points_file = f"batch_{batch_number}_depth_points_{nn}.bin"
        confidence_points_file = f"batch_{batch_number}_confidence_points_{nn}.bin"
        depth_map_camera_file = f"batch_{batch_number}_depth_map_camera_{mm}.bin"
        depth_map_colour_file = f"batch_{batch_number}_depth_map_colour_{mm}.bin"
        depth_map_grey_file = f"batch_{batch_number}_depth_map_grey_{mm}.bin"
        tracked_point_file = f"batch_{batch_number}_tracked_point_{mm}.bin"
        extrinsic_matrix_file = f"batch_{batch_number}_extrinsic_matrix_{mm}.bin"
        texture_intrinsics_file = f"batch_{batch_number}_texture_intrinsics_{mm}.bin"
        camera_intrinsics_file = f"batch_{batch_number}_camera_intrinsics_{mm}.bin"

        depth_points_path = os.path.join(directory, depth_points_file)
        confidence_points_path = os.path.join(directory, confidence_points_file)
        depth_map_camera_path = os.path.join(directory, depth_map_camera_file)
        depth_map_colour_path = os.path.join(directory, depth_map_colour_file)
        depth_map_grey_path = os.path.join(directory, depth_map_grey_file)
        tracked_point_path = os.path.join(directory, tracked_point_file)
        extrinsic_matrix_path = os.path.join(directory, extrinsic_matrix_file)
        texture_intrinsics_path = os.path.join(directory, texture_intrinsics_file)
        camera_intrinsics_path = os.path.join(directory, camera_intrinsics_file)

        depth_points_exists = depth_points_file if os.path.exists(depth_points_path) else "n/a"
        confidence_points_exists = confidence_points_file if os.path.exists(confidence_points_path) else "n/a"
        depth_map_camera_exists = depth_map_camera_file if os.path.exists(depth_map_camera_path) else "n/a"
        depth_map_colour_exists = depth_map_colour_file if os.path.exists(depth_map_colour_path) else "n/a"
        depth_map_grey_exists = depth_map_grey_file if os.path.exists(depth_map_grey_path) else "n/a"
        tracked_point_exists = tracked_point_file if os.path.exists(tracked_point_path) else "n/a"
        extrinsic_matrix_exists = extrinsic_matrix_file if os.path.exists(extrinsic_matrix_path) else "n/a"
        texture_intrinsics_exists = texture_intrinsics_file if os.path.exists(texture_intrinsics_path) else "n/a"
        camera_intrinsics_exists = camera_intrinsics_file if os.path.exists(camera_intrinsics_path) else "n/a"

        filename_table.append([depth_points_exists, depth_map_camera_exists, depth_map_colour_exists, depth_map_grey_exists, confidence_points_exists, tracked_point_exists, extrinsic_matrix_exists, texture_intrinsics_exists, camera_intrinsics_exists])

    return filename_table

    
#######################################################################################################################
### extracting the data from the files 


def read_float_data_as_nxm(file_path, file_name, confidence_level=None, m=4):
    """
    Reads binary float32 data from a file and reshapes it into an N x M NumPy array.
    The function loads raw binary data (little-endian float32) from the specified file,
    reshapes it into a 2D array with `m` columns, and optionally filters out rows
    where the confidence value (assumed to be in the last column) is below the given threshold.
    Parameters:
        file_path (str): Path to the directory containing the file.
        file_name (str): Name of the binary file to read.
        confidence_level (float, optional): Minimum confidence value required to keep a row.
            If None, no filtering is applied. Default is None.
        m (int, optional): Number of columns to reshape the data into. Default is 4.
    Returns:
        np.ndarray: A 2D NumPy array of shape (N, m) containing the filtered data.
    """

    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # Reshape the data into a 2D array with 4 columns
    # x, y, depth, conf
    data_reshaped = data.reshape(-1, m)

    # Filter out points with confidence below the specified level
    if confidence_level is not None:
        data_reshaped = data_reshaped[data_reshaped[:, m-1] >= confidence_level]
    return data_reshaped


def read_float_data_as_nx6(file_path, file_name):
    """
    Reads a binary file containing float32 data and reshapes it into a 2D NumPy array with 6 columns.
    Parameters:
        file_path (str): The directory path to the binary file.
        file_name (str): The name of the binary file to read.
    Returns:
        numpy.ndarray: A 2D array of shape (n, 6), where n is determined by the total number of float32 values divided by 6.
    Notes:
        - The binary file is expected to contain little-endian float32 values.
        - The total number of float32 values in the file must be a multiple of 6.
    """
    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # Reshape the data into a 2D array with 6 columns
    data_reshaped = data.reshape(-1, 6)

    return data_reshaped

def get_depth_map_bitmap(data, depth_points=None, tracked_points=None, depth_range=(0.0, 5.0), colour_map='inferno', rotation=90):
    """
    Generates a bitmap image representing a depth map with optional overlays for depth points and tracked points.
    Parameters
    ----------
    data : array-like or None
        An array of pixel data, where each row should contain (x, y, a, r, g, b).
        If None, a uniform background is used.
    depth_points : np.ndarray or None, optional
        Array of depth points to overlay, with shape (N, 3) where columns are (x, y, depth).
        Points are colored according to their normalized depth using the specified colour_map.
    tracked_points : np.ndarray or None, optional
        Array of tracked points to overlay, with shape (M, 2) where columns are (x, y).
        Points are highlighted as blue circles.
    depth_range : tuple of float, optional
        The minimum and maximum depth values for normalization (default is (0.0, 5.0)).
    colour_map : str, optional
        The name of the matplotlib colormap to use for depth points (default is 'inferno').
    rotation : int, optional
        The rotation angle (in degrees) to apply to the final image (default is 90).
    Returns
    -------
    depth_map_bitmap : np.ndarray
        The resulting RGB bitmap image as a NumPy array of shape (height, width, 3) and dtype uint8.
    """

    num_cols = 640
    num_rows = 480

    depth_min = depth_range[0]
    depth_max = depth_range[1]

    # If data is None, return a uniform pale blue background
    if colour_map == 'inferno':
        rgb_img = np.full((num_rows, num_cols, 3), [1.0, 1.0, 1.0], dtype=np.float32)
    else:
        rgb_img = np.full((num_rows, num_cols, 3), [0.0, 0.0, 0.0], dtype=np.float32)

    if data is not None:
        for row in data:
            x, y, a, r, g, b = row
            xi, yi = int(x), int(y)
            if 0 <= xi < num_cols and 0 <= yi < num_rows:
                rgb_img[yi, xi, 0] = r
                rgb_img[yi, xi, 1] = g
                rgb_img[yi, xi, 2] = b

        # If values are in 0-255, normalize to 0-1
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0

    # # Overlay points if provided
    if depth_points is not None and len(depth_points) > 0:
        xs = depth_points[:, 0].astype(int)
        ys = depth_points[:, 1].astype(int)
        values = depth_points[:, 2]

        # Normalize values to [0, 1] for color mapping
        norm_values = 1.0 - (values - depth_min) / (depth_max - depth_min + 1e-8)
        norm_values = np.clip(norm_values, 0.0, 1.0)
        cmap = plt.get_cmap(colour_map)
        colors = cmap(norm_values)[:, :3]  # Get RGB colors

        for x, y, color in zip(xs, ys, colors):
            if 0 <= x < num_cols and 0 <= y < num_rows:
                # Draw a 3x3 square centered at (y, x)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < num_cols and 0 <= ny < num_rows:
                            rgb_img[ny, nx] = color

    # Tracked points if provided
    if tracked_points is not None and len(tracked_points) > 0:
        xs = tracked_points[:, 0].astype(int)  # because of different coordinate systems
        ys = tracked_points[:, 1].astype(int)

        # Draw a blue circle centered at each tracked point
        radius = 20
        thickness = 4

        for x, y in zip(xs, ys):
            if 0 <= x < num_cols and 0 <= y < num_rows:
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if radius - thickness <= dist <= radius + 0.5:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < num_cols and 0 <= ny < num_rows:
                                rgb_img[ny, nx] = [1.0, 1.0, 1.0]

    # Convert to uint8 bitmap

    rotated_img = rotate_rgb_img(rgb_img, rotation)
    depth_map_bitmap = (rotated_img * 255).astype(np.uint8)
    return depth_map_bitmap

def rotate_rgb_img(rgb_img, rotation_deg):
    """
    Rotates an RGB image by a specified degree.
    Parameters:
        rgb_img (np.ndarray): The input RGB image as a NumPy array.
        rotation_deg (int): The rotation angle in degrees. Must be one of {0, 90, -90, 180, 270}.
    Returns:
        np.ndarray: The rotated RGB image.
    Raises:
        ValueError: If rotation_deg is not one of {0, 90, -90, 180, 270}.
    """
 
    r = rotation_deg % 360
    if r == 0:
        return rgb_img
    elif r == 90:           # clockwise 90°
        return np.rot90(rgb_img, k=3)  # rot90 is counter-clockwise
    elif r == 180:
        return np.rot90(rgb_img, k=2)
    elif r == 270 or rotation_deg == -90:
        return np.rot90(rgb_img, k=1)
    else:
        raise ValueError("rotation_deg must be one of {0, 90, -90, 180, 270}")

def get_points_and_images_bitmaps(file_path, row, confidence_level, depth_range):
    """
    Extracts and generates bitmap images representing various depth and confidence maps from data files.
    Parameters:
        file_path (str): Path to the data file containing depth and image information.
        row (list): List of strings representing file keys or identifiers for different data types.
        confidence_level (float): Confidence threshold for filtering depth points.
        depth_range (tuple): Minimum and maximum depth values for visualization.
    Returns:
        tuple:
            depth_points_bitmap: Bitmap image of depth points.
            depth_map_camera_bitmap: Bitmap image of the camera depth map.
            depth_map_colour_bitmap: Bitmap image of the colored depth map.
            confidence_points_bitmap: Bitmap image of confidence points.
    Notes:
        - If a particular data type is not available (indicated by "n/a" in the row), a default or empty bitmap is returned for that type.
        - The function internally reads and processes data using helper functions such as `read_float_data_as_nxm`, `read_float_data_as_nx6`, and `get_depth_map_bitmap`.
        - The returned bitmaps can be used for visualization or further analysis.
    """

    depth_points_bitmap = None
    depth_map_camera_bitmap = None
    depth_map_colour_bitmap = None
    confidence_points_bitmap = None

    overlay_points = None


    # DEPTH POINTS
    if row[0] != "n/a":
        points = read_float_data_as_nxm(file_path, row[0], confidence_level)
        depth_points_bitmap = get_depth_map_bitmap(data=None, depth_points=points, depth_range=depth_range)
    else:
        depth_points_bitmap = get_depth_map_bitmap(data=None, depth_points=None)

    # CAMERA IMAGE
    if row[1] != "n/a":
        points = read_float_data_as_nxm(file_path, row[0])
        depth_map_camera = read_float_data_as_nx6(file_path, row[1])
        if row[5] != "n/a":
            tracked_points = read_float_data_as_nxm(file_path, row[5])
            depth_map_camera_bitmap = get_depth_map_bitmap(depth_map_camera, depth_points=points, tracked_points=tracked_points, depth_range=depth_range)
        else:
            depth_map_camera_bitmap = get_depth_map_bitmap(depth_map_camera, depth_points=points, depth_range=depth_range)
    else:
        depth_map_camera_bitmap = get_depth_map_bitmap(None)

    # DEPTH MAP
    if row[3] != "n/a":
        depth_map = read_float_data_as_nx6(file_path, row[3])
        depth_map_colour_bitmap = get_depth_map_bitmap(depth_map, overlay_points, depth_range=depth_range)
    else:
        depth_map_colour_bitmap = get_depth_map_bitmap(None)

    # CONFIDENCE POINTS
    if row[4] != "n/a":
        confidence_points = read_float_data_as_nxm(file_path, row[4])
        confidence_points_bitmap = get_depth_map_bitmap(data=None, depth_range=(0.0, 1.0), depth_points=confidence_points, colour_map='Greys')
        overlay_points = confidence_points
    else:
        confidence_points_bitmap = get_depth_map_bitmap(data=None, depth_points=None)

    return depth_points_bitmap, depth_map_camera_bitmap, depth_map_colour_bitmap, confidence_points_bitmap

def get_depth_point_vs_map_data(file_path, depth_points_file, depth_map_file, confidence_level,
                                width=640, height=480, width_crop_size=0, height_crop_size=0):
    """
    Loads depth point data and corresponding per-pixel depth map data, then combines them by mapping each point to its
    corresponding pixel value in the depth map.
    Args:
        file_path (str): Path to the directory containing the data files.
        depth_points_file (str): Filename of the depth points data file.
        depth_map_file (str): Filename of the depth map data file.
        confidence_level (float): Minimum confidence threshold for filtering depth points.
        width (int, optional): Width of the depth map image. Defaults to 640.
        height (int, optional): Height of the depth map image. Defaults to 480.
        width_crop_size (int, optional): Number of pixels to crop from the left and right edges. Defaults to 0.
        height_crop_size (int, optional): Number of pixels to crop from the top and bottom edges. Defaults to 0.
    Returns:
        tuple:
            - combined (np.ndarray): Array of shape (N, 5) containing [x, y, depth, conf, map_value] for each point.
                If a point is out of bounds, map_value is NaN.
            - depth_map (np.ndarray): Array of per-pixel depth map data with shape (width * height, 6).
    """

    # Load [x, y, depth, conf]
    depth_points = read_float_data_as_nxm(file_path, depth_points_file, confidence_level)
    if depth_points.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    # Output buffer
    num_points = depth_points.shape[0]
    combined = np.empty((num_points, 5), dtype=np.float32)
    combined[:, :4] = depth_points[:, :4]
    combined[:, 4] = np.nan  # default: NaN if no valid map lookup

    # Load per-pixel depth map data [x, y, a, r, g, b] flattened row-major
    depth_map = read_float_data_as_nx6(file_path, depth_map_file)

    # Integer pixel coords (nearest-neighbour). Use truncation for speed.
    xi = depth_points[:, 0].astype(np.int32)
    yi = depth_points[:, 1].astype(np.int32)

    # In-bounds mask
    inbounds = (xi >= 0 + width_crop_size) & (xi < width - width_crop_size) & (yi >= 0 + height_crop_size) & (yi < height - height_crop_size)
    if not np.any(inbounds):
        return combined

    # Row-major flat indices
    index = yi[inbounds] * width + xi[inbounds]

    # Use r channel only (greyscale assumption r=g=b)
    combined[inbounds, 4] = depth_map[index, 3].astype(np.float32)

    return combined, depth_map




#######################################################################################################################
### display & visualisation

def plot_points_and_images(depth_points_bitmap, depth_map_camera_bitmap, depth_map_colour_bitmap, confidence_points_bitmap, rotation_deg):
    """
    Displays four bitmap images side by side after rotating them by a specified angle.
    Parameters:
        depth_points_bitmap (np.ndarray): Bitmap image representing depth points.
        depth_map_camera_bitmap (np.ndarray): Bitmap image representing the depth map from the camera.
        depth_map_colour_bitmap (np.ndarray): Bitmap image representing the coloured depth map.
        confidence_points_bitmap (np.ndarray): Bitmap image representing confidence points.
        rotation_deg (int): Angle in degrees to rotate each image. Must be one of {0, 90, -90, 180, 270}.
    Raises:
        ValueError: If rotation_deg is not one of the allowed values.
    Displays:
        A matplotlib figure with the four rotated images side by side, each with a corresponding title.
    """
    # Rotate all images by the specified angle
    def rotate_img(img, rotation_deg):
        if img is None:
            return img
        r = rotation_deg % 360
        if r == 0:
            return img
        elif r == 90:
            return np.rot90(img, k=3)
        elif r == 180:
            return np.rot90(img, k=2)
        elif r == 270 or rotation_deg == -90:
            return np.rot90(img, k=1)
        else:
            raise ValueError("rotation_deg must be one of {0, 90, -90, 180, 270}")

    depth_points_bitmap = rotate_img(depth_points_bitmap, rotation_deg)
    depth_map_camera_bitmap = rotate_img(depth_map_camera_bitmap, rotation_deg)
    depth_map_colour_bitmap = rotate_img(depth_map_colour_bitmap, rotation_deg)
    confidence_points_bitmap = rotate_img(confidence_points_bitmap, rotation_deg)

    # Display the four bitmaps side by side
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    axes[0].imshow(depth_points_bitmap)
    axes[0].set_title('Depth Points Bitmap')
    axes[1].imshow(depth_map_camera_bitmap)
    axes[1].set_title('Depth Map Camera Bitmap')
    axes[2].imshow(depth_map_colour_bitmap)
    axes[2].set_title('Depth Map Colour Bitmap')
    axes[3].imshow(confidence_points_bitmap)
    axes[3].set_title('Confidence Points Bitmap')
    plt.tight_layout()
    plt.show()

def batch_display_points_and_images(file_path, collated_table, confidence_level, timestamps_array=None, match_indices=None, depth_points_indices=None, depth_range=(0.0, 5.0), rotation_deg=0):
    """
    Displays depth points and associated images in batches, filtering and visualizing data based on provided indices and confidence levels.
    Args:
        file_path (str): Path to the directory containing the data files.
        collated_table (list): Table containing rows of metadata and file references for depth points and images.
        confidence_level (float): Minimum confidence threshold for displaying points.
        timestamps_array (np.ndarray, optional): Array containing timestamps for frames, cameras, depth, and confidence. Defaults to None.
        match_indices (list, optional): List of index pairs mapping rows to timestamp entries. Defaults to None.
        depth_points_indices (list, optional): List of indices specifying which depth points to process. Defaults to None.
        depth_range (tuple, optional): Range of depth values to filter points, as (min_depth, max_depth). Defaults to (0.0, 5.0).
        rotation_deg (int, optional): Degree of rotation to apply to displayed images. Defaults to 0.
    Returns:
        None
    Side Effects:
        - Displays images and points using visualization functions.
        - Prints row information and timestamp details for each processed entry.
    """

    for i_row, row in enumerate(collated_table):


        depth_points_file = row[0]
        if depth_points_file == "n/a":
            continue
        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])

        if (depth_points_index not in depth_points_indices):
            continue

        indices = match_indices[i_row]

        depth_points_bitmap, depth_map_camera_bitmap, depth_map_grey_bitmap, confidence_points_bitmap = get_points_and_images_bitmaps(file_path, row, confidence_level, depth_range)

        plot_points_and_images(depth_points_bitmap, depth_map_camera_bitmap, depth_map_grey_bitmap, confidence_points_bitmap, rotation_deg)

        print(row)
        print(f"ABS: Row {i_row}: Frame ts {timestamps_array[indices[1], 1]:.6f}, Camera ts {timestamps_array[indices[1], 2]:.6f}, Depth ts {timestamps_array[indices[0], 3]:.6f}, Confidence ts {timestamps_array[indices[1], 4]:.6f}")
        print(f"DEL: Row {i_row}: Frame ts {timestamps_array[indices[1], 1]-timestamps_array[indices[0], 3]:.6f}, Camera ts {timestamps_array[indices[1], 2]-timestamps_array[indices[0], 3]:.6f}, Depth ts n/a, Confidence ts {timestamps_array[indices[1], 4]-timestamps_array[indices[0], 3]:.6f}")


def plot_histograms_and_regression(combined_data_points, depth_range, depth_map, regression_points = None, flip_axes = False):
    """
    Plots histograms and a scatter plot to visualize depth data and regression results.
    This function creates a 1x4 subplot figure:
        1. Histogram of valid reciprocal relative depth values from the depth map.
        2. Histogram of reciprocal relative depth values from combined data points.
        3. Histogram of metric depth values from combined data points.
        4. Scatter plot of metric depth vs reciprocal relative depth, optionally with a regression line.
    Parameters
    ----------
    combined_data_points : np.ndarray
        Array of data points, where columns are expected to include metric depth (col 2),
        weights (col 3), and reciprocal relative depth (col 4).
    depth_range : tuple
        Tuple (min_depth, max_depth) specifying the range of metric depth values.
    depth_map : np.ndarray
        Array representing the depth map, where column 4 contains reciprocal relative depth values.
    regression_points : np.ndarray, optional
        Array of regression points to plot as a line in the scatter plot (default is None).
    flip_axes : bool, optional
        If True, flips the axes of the histograms and scatter plot (default is False).
    Returns
    -------
    None
        Displays the plots using matplotlib.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    depth_min = depth_range[0]
    depth_max = depth_range[1]
    depth_map_min = 0.0
    depth_map_max = 1.0

    # Histogram of Depth Values
    # Only include points with values > 0
    valid_values = depth_map[:, 4][depth_map[:, 4] > 0]
    hist_max = np.histogram(valid_values, bins=50)[0].max()
    axes[0].hist(valid_values, bins=50, alpha=0.7, color='blue')
    axes[0].set_xlim(depth_map_min, depth_map_max)
    axes[0].set_xlabel('Reciprocal Relative Depth')
    axes[0].set_ylim(0, hist_max)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True)

    hist_max = np.histogram(combined_data_points[:, 2], bins=50)[0].max()
    # if not flip_axes:
    #     axes[1].hist(combined_data_points[:, 2], bins=50, alpha=0.7, color='blue')
    #     axes[1].set_xlim(depth_min, depth_max)
    #     axes[1].set_xlabel('Metric Depth')
    # else:
    axes[1].hist(combined_data_points[:, 4], bins=50, alpha=0.7, color='blue')
    axes[1].set_xlim(depth_map_min, depth_map_max)
    axes[1].set_xlabel('Reciprocal Relative Depth')
    axes[1].set_ylim(0, hist_max)
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)

    # Histogram of Depth Map Values
    # if not flip_axes:
    #     axes[2].hist(combined_data_points[:, 4], bins=50, alpha=0.7, color='blue')
    #     axes[2].set_xlim(depth_map_min, depth_map_max)
    #     axes[2].set_xlabel('Reciprocal Relative Depth')
    # else:
    axes[2].hist(combined_data_points[:, 2], bins=50, alpha=0.7, color='blue')
    axes[2].set_xlim(depth_min, depth_max)
    axes[2].set_xlabel('Metric Depth')
    axes[2].set_ylim(0, hist_max)
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True)

    # Scatter plot: Depth vs Depth Map Value
    weights = combined_data_points[:, 3]
    min_size, max_size = 2, 50
    norm_weights = (weights - weights.min()) / (weights.ptp() + 1e-8)
    sizes = min_size + norm_weights * (max_size - min_size)

    if regression_points is not None:
        if not flip_axes:
            axes[3].plot(regression_points[:, 0], regression_points[:, 1], linestyle='--', color='blue', alpha=0.7, linewidth=2, label='Regression')
        else:
            axes[3].plot(regression_points[:, 1], regression_points[:, 0], linestyle='--', color='blue', alpha=0.7, linewidth=2, label='Regression')
        axes[3].legend()
    if not flip_axes:
        axes[3].scatter(combined_data_points[:, 2], combined_data_points[:, 4], alpha=0.5, s=sizes)
        axes[3].set_xlim(0.0, depth_max)
        axes[3].set_ylim(0.0, depth_map_max)
        axes[3].set_xlabel('Depth')
        axes[3].set_ylabel('Depth Map Value')
    else:
        axes[3].scatter(combined_data_points[:, 4], combined_data_points[:, 2], alpha=0.5, s=sizes)
        axes[3].set_xlim(0.0, depth_map_max)
        axes[3].set_ylim(0.0, depth_max)
        axes[3].set_xlabel('Depth Map Value')
        axes[3].set_ylabel('Depth')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

def batch_display_histograms_and_regression(file_path, matched_filename_table, depth_points_indices, confidence_level, depth_range=(0.0, 5.0), flip_axes=False):
    """
    Processes a batch of depth point and depth map files, displaying histograms and regression plots for each valid entry.
    Iterates over the provided matched filename table, filtering entries based on availability and specified indices.
    For each valid entry, extracts combined data points and visualizes them using histograms and regression analysis.
    Args:
        file_path (str): Path to the directory containing the data files.
        matched_filename_table (list): List of rows, each containing filenames and metadata for depth points and depth maps.
        depth_points_indices (list): List of integer indices specifying which depth points files to process.
        confidence_level (float): Minimum confidence threshold for including data points.
        depth_range (tuple, optional): Range of depth values to display in plots. Defaults to (0.0, 5.0).
        flip_axes (bool, optional): If True, flips the axes in the plots. Defaults to False.
    Returns:
        None
    """

    for _, row in enumerate(matched_filename_table):
        
        depth_points_file = row[0]
        depth_map_file = row[3]

        # do nothing if the depth points file is not available
        if depth_points_file == "n/a":
            continue

        # make sure the depth points index is in the list of ones to do (enables us to skip some)
        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])
        if (depth_points_index not in depth_points_indices):
            continue

        print(row)

        # combined data points: x, y, dpeth, confidence, depth map value
        combined_data_points = get_depth_point_vs_map_data(file_path, depth_points_file, depth_map_file, confidence_level)

        plot_histograms_and_regression(combined_data_points, depth_range, flip_axes)


####################################################################################################################


if (__name__ == "__main__"):

    ##########################################################################################################

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # DATA - A
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
    # DATA - B
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
    # DATA - C
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

    ###########################################################################################################
    
    BATCH_NUMBER = 0
    CONFIDENCE_LEVEL = 0.75
    DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 25.0)

    DEPTH_POINTS_INDICES = range(0, 10, 1)

    MATCH_TIMESTAMPS = False

    # Apply a sigmoid weight dropping towards zero above 7, width 1
    X_CUT = 7.0
    X_WIDTH = 1.0
    WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

    FLIP_AXES = False

    ##################################################################################################################
    TIMESTAMPS_TABLE = read_timestamp_files(FILE_PATH, BATCH_NUMBER)

    if MATCH_TIMESTAMPS:
        MATCHED_INDICES = find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
    else:
        MATCHED_INDICES = get_all_indices(FILE_PATH, BATCH_NUMBER)

    print_closest_ts_match(TIMESTAMPS_TABLE, MATCHED_INDICES)
    FILENAME_TABLE = get_matched_filenames(MATCHED_INDICES, FILE_PATH, BATCH_NUMBER)

    depth_map_file_name_replacement = None
    batch_display_points_and_images(FILE_PATH, FILENAME_TABLE, CONFIDENCE_LEVEL, TIMESTAMPS_TABLE, MATCHED_INDICES, DEPTH_POINTS_INDICES, depth_range=DEPTH_RANGE_FOR_COLOUR_MAP)

    batch_display_histograms_and_regression(FILE_PATH, FILENAME_TABLE, DEPTH_POINTS_INDICES, CONFIDENCE_LEVEL, depth_range=DEPTH_RANGE_FOR_COLOUR_MAP, flip_axes=FLIP_AXES)


####################################################################################################################
