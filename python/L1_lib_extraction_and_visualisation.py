import os
import re
import numpy as np
import struct
import matplotlib.pyplot as plt

########################################################################################################################
### extracting the time stamps and then matching depth images with camera images

def read_timestamp_files(directory, batch_number):
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
    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # Reshape the data into a 2D array with 6 columns
    data_reshaped = data.reshape(-1, 6)

    return data_reshaped

def get_depth_map_bitmap(data, depth_points=None, tracked_points=None, depth_range=(0.0, 5.0), colour_map='inferno', rotation=90):

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
    Rotate an RGB image array (H, W, 3) by multiples of 90 degrees.
    rotation_deg can be 0, 90, -90, 180, or 270.
    Returns a new rotated array.
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
    Returns N x 5 array: [x, y, depth, confidence, depth_map_value]
    Assumes depth_map_data rows are [x, y, a, r, g, b] and r=g=b (greyscale).
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

    return combined




#######################################################################################################################
### display & visualisation

def plot_points_and_images(depth_points_bitmap, depth_map_camera_bitmap, depth_map_colour_bitmap, confidence_points_bitmap, rotation_deg):
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


def plot_histograms_and_regression(combined_data_points, depth_range, regression_points = None):

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        depth_min = depth_range[0]
        depth_max = depth_range[1]
        depth_map_min = 0.0
        depth_map_max = 1.0
        hist_max = 100

        # Histogram of Depth Values
        axes[0].hist(combined_data_points[:, 2], bins=50, alpha=0.7, color='blue')
        axes[0].set_xlim(depth_min, depth_max)
        axes[0].set_ylim(0, hist_max)
        axes[0].set_xlabel('Depth')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Histogram of Depth Values')
        axes[0].grid(True)

        # Histogram of Depth Map Values
        axes[1].hist(combined_data_points[:, 4], bins=50, alpha=0.7, color='blue')
        axes[1].set_xlim(depth_map_min, depth_map_max)
        axes[1].set_ylim(0, hist_max)
        axes[1].set_xlabel('Depth Map Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Histogram of Depth Map Values')
        axes[1].grid(True)

        # Scatter plot: Depth vs Depth Map Value
        # Use combined_data_points[:, 3] (confidence) as weights for point size
        # Normalize weights to a reasonable range, e.g., [2, 50]
        weights = combined_data_points[:, 3]
        min_size, max_size = 2, 50
        norm_weights = (weights - weights.min()) / (weights.ptp() + 1e-8)
        sizes = min_size + norm_weights * (max_size - min_size)

        if regression_points is not None:
            axes[2].plot(regression_points[:, 0], regression_points[:, 1], linestyle='--', color='blue', alpha=0.7, linewidth=2, label='Regression')
            axes[2].legend()
        axes[2].scatter(combined_data_points[:, 2], combined_data_points[:, 4], alpha=0.5, s=sizes)
        axes[2].set_xlim(0.0, depth_max)
        axes[2].set_ylim(0.0, depth_map_max)
        axes[2].set_xlabel('Depth')
        axes[2].set_ylabel('Depth Map Value')
        axes[2].set_title('Depth Map Value vs Depth')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

def batch_display_histograms_and_regression(file_path, matched_filename_table, depth_points_indices, confidence_level, depth_range=(0.0, 5.0)):

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

        plot_histograms_and_regression(combined_data_points, depth_range)


####################################################################################################################


if (__name__ == "__main__"):

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"

    BATCH_NUMBER = 0
    CONFIDENCE_LEVEL = 0.75
    DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 25.0)

    DEPTH_POINTS_INDICES = range(0, 10, 1)

    MATCH_TIMESTAMPS = False

    # Apply a sigmoid weight dropping towards zero above 7, width 1
    X_CUT = 7.0
    X_WIDTH = 1.0
    WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

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

    batch_display_histograms_and_regression(FILE_PATH, FILENAME_TABLE, DEPTH_POINTS_INDICES, CONFIDENCE_LEVEL, depth_range=DEPTH_RANGE_FOR_COLOUR_MAP)


####################################################################################################################
