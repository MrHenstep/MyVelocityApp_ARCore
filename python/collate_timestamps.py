import os
import re
import numpy as np
import struct
import matplotlib.pyplot as plt

####################################################################################################################

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

def find_closest_rows(array, search_col, target_col, direction='both'):
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
    search_values = array[:, search_col]
    target_values = array[:, target_col]

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
   
    # print("\nClosest Matches ...\n")
    # for i, indices in enumerate(match_indices):

    #     print(f"Row {i}: Frame ts {timestamps_array[indices[1], 1]:.6f}, Camera ts {timestamps_array[indices[1], 2]:.6f}, Depth ts {timestamps_array[indices[0], 3]:.6f}, Confidence ts {timestamps_array[indices[1], 4]:.6f}")

    print("\nClosest Matches difference ...\n")
    for i, indices in enumerate(match_indices):

        print(f"Row {i}: diff, Frame ts {timestamps_array[indices[1], 1]-timestamps_array[indices[0], 3]:.6f}, Camera ts {timestamps_array[indices[1], 2]-timestamps_array[indices[0], 3]:.6f}, Depth ts {timestamps_array[indices[1], 3]-timestamps_array[indices[0], 3]:.6f}, Confidence ts {timestamps_array[indices[1], 4]-timestamps_array[indices[0], 3]:.6f}")

    # print("\nStatistics on differences vs original depth ts...")

    # diffs = timestamps_array[match_indices[:, 1], 1] - timestamps_array[match_indices[:, 0], 3]
    # print(f"Frame: Mean: {np.mean(diffs):.6f}, Std: {np.std(diffs):.6f}, Min: {np.min(diffs):.6f}, Max: {np.max(diffs):.6f}")

    # diffs = timestamps_array[match_indices[:, 1], 2] - timestamps_array[match_indices[:, 0], 3]
    # print(f"Camera: Mean: {np.mean(diffs):.6f}, Std: {np.std(diffs):.6f}, Min: {np.min(diffs):.6f}, Max: {np.max(diffs):.6f}")

    # diffs = timestamps_array[match_indices[:, 1], 2] - timestamps_array[match_indices[:, 0], 3]
    # print(f"Depth: Mean: {np.mean(diffs):.6f}, Std: {np.std(diffs):.6f}, Min: {np.min(diffs):.6f}, Max: {np.max(diffs):.6f}")

    # diffs = timestamps_array[match_indices[:, 1], 4] - timestamps_array[match_indices[:, 0], 3]
    # print(f"Confidence: Mean: {np.mean(diffs):.6f}, Std: {np.std(diffs):.6f}, Min: {np.min(diffs):.6f}, Max: {np.max(diffs):.6f}")

def get_matched_filenames(match_indices, directory, batch_number):


    collated_table = []
    # directory = "../exported"

    for idx_pair in match_indices:
        nn = idx_pair[0]
        mm = idx_pair[1]
        depth_points_file = f"batch_{batch_number}_depth_points_{nn}.bin"
        confidence_points_file = f"batch_{batch_number}_confidence_points_{nn}.bin"
        depth_map_camera_file = f"batch_{batch_number}_depth_map_camera_{mm}.bin"
        depth_map_colour_file = f"batch_{batch_number}_depth_map_colour_{mm}.bin"
        depth_map_grey_file = f"batch_{batch_number}_depth_map_grey_{mm}.bin"

        depth_points_path = os.path.join(directory, depth_points_file)
        confidence_points_path = os.path.join(directory, confidence_points_file)
        depth_map_camera_path = os.path.join(directory, depth_map_camera_file)
        depth_map_colour_path = os.path.join(directory, depth_map_colour_file)
        depth_map_grey_path = os.path.join(directory, depth_map_grey_file)

        depth_points_exists = depth_points_file if os.path.exists(depth_points_path) else "n/a"
        confidence_points_exists = confidence_points_file if os.path.exists(confidence_points_path) else "n/a"
        depth_map_camera_exists = depth_map_camera_file if os.path.exists(depth_map_camera_path) else "n/a"
        depth_map_colour_exists = depth_map_colour_file if os.path.exists(depth_map_colour_path) else "n/a"
        depth_map_grey_exists = depth_map_grey_file if os.path.exists(depth_map_grey_path) else "n/a"

        collated_table.append([depth_points_exists, depth_map_camera_exists, depth_map_colour_exists, depth_map_grey_exists, confidence_points_exists])

    return collated_table


def get_depth_points_data(file_path, file_name, confidence_level):

    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # Reshape the data into a 2D array with 4 columns
    # x, y, depth, conf
    data_reshaped = data.reshape(-1, 4)

    # Filter out points with confidence below the specified level
    data_reshaped = data_reshaped[data_reshaped[:, 3] >= confidence_level]
    return data_reshaped 

# def get_depth_points_bitmap(overlay_points=None):
    num_rows = 640
    num_cols = 480

    # Create a uniform grey background (e.g., 0.5 for all channels)
    rgb_img = np.full((num_rows, num_cols, 3), 1.0, dtype=np.float32)
    
    # Overlay points if provided
    if overlay_points is not None and len(overlay_points) > 0:
        # overlay_points should be an array of shape (N, 3): x, y, value
        xs = overlay_points[:, 1].astype(int)
        ys = overlay_points[:, 0].astype(int)
        values = overlay_points[:, 2]

        # Normalize values to [0, 1] for color mapping
        norm_values = (values - np.min(values)) / (np.ptp(values) + 1e-8)
        cmap = plt.get_cmap('inferno')
        colors = cmap(norm_values)[:, :3]  # Get RGB colors

        for x, y, color in zip(xs, ys, colors):
            if 0 <= x < num_cols and 0 <= y < num_rows:
                # Draw a 3x3 square centered at (y, x)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < num_rows and 0 <= ny < num_cols:
                            rgb_img[ny, num_cols - nx] = color

    # Convert to uint8 bitmap
    depth_points_bitmap = (rgb_img * 255).astype(np.uint8)
    return depth_points_bitmap

def get_depth_map_data(file_path, file_name):
    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # Reshape the data into a 2D array with 6 columns
    data_reshaped = data.reshape(-1, 6)

    return data_reshaped

def get_depth_map_bitmap(data, overlay_points=None, depth_range=(0.0, 5.0), colour_map='inferno'):
    num_rows = 640
    num_cols = 480

    depth_min = depth_range[0]
    depth_max = depth_range[1]

    # If data is None, return a uniform pale blue background
    if colour_map == 'inferno':
        rgb_img = np.full((num_rows, num_cols, 3), [1.0, 1.0, 1.0], dtype=np.float32)
    else:
        rgb_img = np.full((num_rows, num_cols, 3), [0.0, 0.0, 0.0], dtype=np.float32)

    if data is not None:
        for row in data:
            y, x, a, r, g, b = row
            xi, yi = int(x), int(y)
            if 0 <= yi < num_cols and 0 <= xi < num_rows:
                rgb_img[xi, yi, 0] = r
                rgb_img[xi, yi, 1] = g
                rgb_img[xi, yi, 2] = b

        # If values are in 0-255, normalize to 0-1
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0

    # Overlay points if provided
    if overlay_points is not None and len(overlay_points) > 0:
        xs = overlay_points[:, 1].astype(int)
        ys = overlay_points[:, 0].astype(int)
        values = overlay_points[:, 2]

        # Normalize values to [0, 1] for color mapping
        # norm_values = 1.0 - (values - np.min(values)) / (np.ptp(values) + 1e-8)
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
                            rgb_img[ny, num_cols - nx] = color

    # Convert to uint8 bitmap
    depth_map_bitmap = (rgb_img * 255).astype(np.uint8)
    return depth_map_bitmap

def display_collated_data(collated_table, confidence_level, timestamps_array=None, match_indices=None, depth_points_indices=None, depth_range=(0.0, 5.0)):

    for i_row, row in enumerate(collated_table):


        depth_points_file = row[0]
        if depth_points_file == "n/a":
            continue
        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])

        if (depth_points_index not in depth_points_indices):
            continue

        print(row)
        indices = match_indices[i_row]

        print(f"ABS: Row {i_row}: Frame ts {timestamps_array[indices[1], 1]:.6f}, Camera ts {timestamps_array[indices[1], 2]:.6f}, Depth ts {timestamps_array[indices[0], 3]:.6f}, Confidence ts {timestamps_array[indices[1], 4]:.6f}")

        print(f"DEL: Row {i_row}: Frame ts {timestamps_array[indices[1], 1]-timestamps_array[indices[0], 3]:.6f}, Camera ts {timestamps_array[indices[1], 2]-timestamps_array[indices[0], 3]:.6f}, Depth ts n/a, Confidence ts {timestamps_array[indices[1], 4]-timestamps_array[indices[0], 3]:.6f}")

        overlay_points = None

        
        # DEPTH POINTS
        if row[0] != "n/a":
            points = get_depth_points_data(file_path, row[0], confidence_level)
            depth_points_bitmap = get_depth_map_bitmap(data=None, overlay_points=points, depth_range=depth_range)
            overlay_points = points
        else:
            depth_points_bitmap = get_depth_map_bitmap(data=None, overlay_points=None)

        # CAMERA IMAGE
        if row[1] != "n/a":
            depth_map_camera = get_depth_map_data(file_path, row[1])
            depth_map_camera_bitmap = get_depth_map_bitmap(depth_map_camera, overlay_points, depth_range=depth_range)
        else:
            depth_map_camera_bitmap = get_depth_map_bitmap(None)

        # DEPTH MAP
        if row[2] != "n/a":
            depth_map = get_depth_map_data(file_path, row[3])
            depth_map_colour_bitmap = get_depth_map_bitmap(depth_map, overlay_points, depth_range=depth_range)
        else:
            depth_map_colour_bitmap = get_depth_map_bitmap(None)

        # CONFIDENCE POINTS
        if row[4] != "n/a":
            confidence_points = get_depth_points_data(file_path, row[4], confidence_level)
            confidence_points_bitmap = get_depth_map_bitmap(data=None, depth_range=(0.0, 1.0), overlay_points=confidence_points, colour_map='Greys')
            overlay_points = confidence_points
        else:
            confidence_points_bitmap = get_depth_map_bitmap(data=None, overlay_points=None)

        # Display the two bitmaps side by side
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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

def histograms_and_regression(file_path, matched_filename_table, depth_points_indices, confidence_level, depth_range=(0.0, 5.0)):

    for i, row in enumerate(matched_filename_table):
        
        depth_points_file = row[0]
        depth_map_file = row[3]


        depth_points_file = row[0]
        if depth_points_file == "n/a":
            continue
        
        depth_points_index = int(depth_points_file.split("_")[-1].split(".")[0])

        if (depth_points_index not in depth_points_indices):
            continue


        # print(depth_map_file)
        print(row)

        # combined data points: x, y, dpeth, confidence, depth map value

        points = get_depth_points_data(file_path, depth_points_file, confidence_level)

        num_points = points.shape[0]

        combined_data_points = np.zeros((num_points, 5), dtype=np.float32)

        combined_data_points[:,:2] = points[:,:2]  # x, y
        combined_data_points[:, 2] = points[:, 2]   # depth
        combined_data_points[:, 3] = points[:, 3]  # confidence

        depth_map_data = get_depth_map_data(file_path, depth_map_file)

        for i_point in range(num_points):
            y, x = combined_data_points[i_point, :2]
            xi, yi = int(x), int(y)
            width=640
            height=480
            index = xi + yi * height
            # print(i_point, (xi, yi, depth_map_data[index, 3]))
            # if 0 <= xi < depth_map_data.shape[0] and 0 <= yi < depth_map_data.shape[1]:
            combined_data_points[i_point, 4] = depth_map_data[index, 3]
            
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
        axes[2].scatter(combined_data_points[:, 2], combined_data_points[:, 4], alpha=0.5)
        axes[2].set_xlim(depth_min, depth_max)
        axes[2].set_ylim(depth_map_min, depth_map_max)
        axes[2].set_xlabel('Depth')
        axes[2].set_ylabel('Depth Map Value')
        axes[2].set_title('Depth Map Value vs Depth')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

####################################################################################################################

cwd = os.getcwd()
print(cwd)

file_path = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported"

# file_path = cwd + "\\..\\exported"
# file_path = "./exported"

batch_number = 1
confidence_level = 0.9
depth_range = (0.0, 15.0)


timestamps_table = read_timestamp_files(file_path, batch_number)
nearest_match_indices = find_closest_rows(timestamps_table, 3, 2, direction='both')

print_closest_ts_match(timestamps_table, nearest_match_indices)
matched_filename_table = get_matched_filenames(nearest_match_indices, file_path, batch_number)


depth_points_indices = range(0, len(timestamps_table))
# depth_points_indices = range(4, 5)

display_collated_data(matched_filename_table, confidence_level, timestamps_table, nearest_match_indices, depth_points_indices, depth_range=depth_range)

histograms_and_regression(file_path, matched_filename_table, depth_points_indices, confidence_level, depth_range=depth_range)


####################################################################################################################

# **** Put points on fixed colour scale
# **** check confidence cut-off
