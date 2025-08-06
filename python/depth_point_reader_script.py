import numpy as np

import matplotlib.pyplot as plt

def show_depth_points(overlay_points=None):
    num_rows = 640
    num_cols = 480

    # Create an empty image (num_rows x num_cols x 3) for RGB
    rgb_img = np.zeros((num_rows, num_cols, 3), dtype=np.float32)

    # for row in data:
    #     y, x, a, r, g, b = row
    #     xi, yi = int(x), int(y)
    #     if 0 <= xi < num_rows and 0 <= yi < num_cols:
    #         rgb_img[xi, yi, 0] = r
    #         rgb_img[xi, yi, 1] = g
    #         rgb_img[xi, yi, 2] = b

    # if rgb_img.max() > 1.0:
    #     rgb_img = rgb_img / 255.0

    plt.imshow(rgb_img)
    plt.title("Depth Map (RGB composite)")
    plt.axis('off')

    # Overlay points if provided
    if overlay_points is not None and len(overlay_points) > 0:
        # overlay_points should be an array of shape (N, 3): x, y, value
        xs = overlay_points[:, 1]
        ys = overlay_points[:, 0]
        values = overlay_points[:, 2]
        plt.scatter(xs, ys, c=values, cmap='inferno', s=2, edgecolors='none')

    plt.show()

file_path = "../exported"
# file_name = "depth_points_10.bin"

# search the folder specified by file_path for the file_name of the form "depth_points_*.bin" and make a list of file names
import os
file_list = [f for f in os.listdir(file_path) if f.startswith("depth_points_") and f.endswith(".bin")]


# now iterate through the file_list
if not file_list:
    print("No files found matching the pattern.")

for file_name in file_list:

    print(f"\nFile: {file_name}")

    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # get number of rows
    num_rows = data.size // 4  # Each row has 4 float32 values
    # print(f"Number of rows: {num_rows}")

    # Reshape the data into a 2D array with 4 columns
    data_reshaped = data.reshape(-1, 4)

    # print out the first 10 rows, neatly formatted with commas in five coloumns, the first column being the index
    num_rows_to_print = 10
    for i, row in enumerate(data_reshaped[:num_rows_to_print]):
        print(f"{i}, {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}")   

    show_depth_points(data_reshaped)

        
for file_name in file_list:

    # Load raw binary float32 data, little-endian
    data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

    # get number of rows
    num_rows = data.size // 4  # Each row has 4 float32 values

    # Reshape the data into a 2D array with 4 columns
    data_reshaped = data.reshape(-1, 4)

    # print out the first 10 rows, neatly formatted with commas in five coloumns, the first column being the index    # Calculate mean depth and confidence
    if num_rows > 0:
        mean_metric_depth = np.mean(data_reshaped[:, 2])
        mean_confidence = np.mean(data_reshaped[:, 3])
        # print file name, number of rows, mean depth, and mean confidence on the same line
        print(f"File: {file_name}, Number of rows: {num_rows}, Mean metric depth: {mean_metric_depth:.6f}, Mean confidence: {mean_confidence:.6f}")
