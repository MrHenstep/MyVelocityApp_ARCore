import numpy as np
import os
import matplotlib.pyplot as plt

def show_depth_map(data, overlay_points=None):
    num_rows = 640
    num_cols = 480

    # Create an empty image (num_rows x num_cols x 3) for RGB
    rgb_img = np.zeros((num_rows, num_cols, 3), dtype=np.float32)

    for row in data:
        y, x, a, r, g, b = row
        xi, yi = int(x), int(y)
        if 0 <= xi < num_rows and 0 <= yi < num_cols:
            rgb_img[xi, yi, 0] = r
            rgb_img[xi, yi, 1] = g
            rgb_img[xi, yi, 2] = b

    if rgb_img.max() > 1.0:
        rgb_img = rgb_img / 255.0

    plt.imshow(rgb_img)
    plt.title("Depth Map (RGB composite)")
    plt.axis('off')

    # Overlay points if provided
    if overlay_points is not None and len(overlay_points) > 0:
        # overlay_points should be an array of shape (N, 3): x, y, value
        xs = overlay_points[:, 1]
        ys = overlay_points[:, 0]
        values = overlay_points[:, 2]
        plt.scatter(xs, ys, c=values, cmap='inferno', s=3, edgecolors='none')

    plt.show()

# fix seed
np.random.seed(42)

num_rows_to_print = 10

file_path = "../exported"
file_number = "3"

file_name_points = "depth_points_" + file_number + ".bin"
file_name_map = "depth_map_colour_" + file_number + ".bin"

###########################################################################################################
# Load the ARCore DEPTH POINTS
###########################################################################################################

print(f"\nFile: {file_name_points}")

# Load raw binary float32 data, little-endian
depth_points = np.fromfile(file_path + "/" + file_name_points, dtype='<f4')  # '<f4' = little-endian float32

# get number of rows
num_rows = depth_points.size // 4  # Each row has 4 float32 values
# print(f"Number of rows: {num_rows}")

# Reshape the data into a 2D array with 4 columns
depth_points_reshaped = depth_points.reshape(-1, 4)

# print out the first 10 rows, neatly formatted with commas in five coloumns, the first column being the index
# indices = np.random.choice(depth_points_reshaped.shape[0], num_rows_to_print, replace=False)
# depth_points_random_rows = depth_points_reshaped[indices]

print(f"\nFile: {file_name_points}")
print(f"Number of rows: {num_rows}")
for i, row in enumerate(depth_points_reshaped[:num_rows_to_print]):
    print(f"{i}, {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}")

mean_metric_depth = np.mean(depth_points_reshaped[:, 2])
mean_confidence = np.mean(depth_points_reshaped[:, 3])
# print file name, number of rows, mean depth, and mean confidence on the same line
print(f"File: {file_name_points}, Number of rows: {num_rows}, Mean metric depth: {mean_metric_depth:.6f}, Mean confidence: {mean_confidence:.6f}")

###########################################################################################################
# Load the Midas DEPTH MAP points (i.e. 640 x 480 of them)
###########################################################################################################

# Load raw binary float32 data, little-endian
depth_map = np.fromfile(file_path + "/" + file_name_map, dtype='<f4')  # '<f4' = little-endian float32

# get number of rows
depth_map_size = depth_map.size
depth_map_num_rows = depth_map_size // 6

print(f"Data size: {depth_map_size}; Number of rows: {depth_map_num_rows}")

# Reshape the data into a 2D array with 6 columns
depth_map_reshaped = depth_map.reshape(-1, 6)
show_depth_map(depth_map_reshaped, overlay_points=depth_points_reshaped)



# print out the first 10 rows, neatly formatted with commas in five coloumns, the first column being the index
# indices = np.random.choice(depth_map_reshaped.shape[0], num_rows_to_print, replace=False)
# depth_map_random_rows = depth_map_reshaped[indices]

print(f"\nFile: {file_name_map}")
print(f"Number of rows: {depth_map_num_rows}")
for i, row in enumerate(depth_map_reshaped[:num_rows_to_print]):
    print(f"{i}, {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}, {row[4]:.6f}, {row[5]:.6f}")


# Calculate mean of the last four columns and print
if depth_map_num_rows > 0:
    mean_a = np.mean(depth_map_reshaped[:, 2])
    mean_r = np.mean(depth_map_reshaped[:, 3])
    mean_g = np.mean(depth_map_reshaped[:, 4])
    mean_b = np.mean(depth_map_reshaped[:, 5])
    # print file name, number of rows, mean a, r, g, b on the same line
    print(f"File: {file_name_map}, Number of rows: {depth_map_num_rows}, Mean a: {mean_a:.6f}, Mean r: {mean_r:.6f}, Mean g: {mean_g:.6f}, Mean b: {mean_b:.6f}")


###########################################################################################################
### Now we can compare the two sets of data
###########################################################################################################

# For each point in the depth points, find the corresponding point in the depth map
# and store in a new array, x, y, depth, confidence, r

depth_points_pairs = []
for point in depth_points_reshaped:
    
    # Find the corresponding point in the depth map
    xx, yy, d, confidence = point
    
    # Convert x, y to integer indices
    x = int(np.round(xx))
    y = int(np.round(yy))

    index = y * 480 + x
    depth_map_point = depth_map_reshaped[index, 2:]

    depth_points_pairs.append((x, y, d, confidence, depth_map_point[1]))
    
    
# Convert to numpy array
depth_points_pairs = np.array(depth_points_pairs)

# Print 10 points selected randomly
print(f"\nDepth Map Points (x, y, depth, confidence, r):")
indices = np.random.choice(depth_points_pairs.shape[0], num_rows_to_print, replace=False)
for i in indices:   
    x, y, d, confidence, r = depth_points_pairs[i]
    print(f"{i}, {x}, {y}, {d:.6f}, {confidence:.6f}, {r:.6f}") 