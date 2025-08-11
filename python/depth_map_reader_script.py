import numpy as np
import os
import matplotlib.pyplot as plt

def show_depth_map(data):
    num_rows = 640
    num_cols = 480

    # Create an empty image (num_rows x num_cols x 3) for RGB
    rgb_img = np.zeros((num_rows, num_cols, 3), dtype=np.float32)

    for row in data:
        y, x, a, r, g, b = row
        xi, yi = int(x), int(y)
        if 0 <= xi < num_rows and 0 <= yi < num_cols:
            # Assign RGB values (assuming r, g, b are in 0-1 or 0-255 range)
            rgb_img[xi, yi, 0] = r
            rgb_img[xi, yi, 1] = g
            rgb_img[xi, yi, 2] = b

    # If values are in 0-255, normalize to 0-1 for imshow
    if rgb_img.max() > 1.0:
        rgb_img = rgb_img / 255.0

    plt.imshow(rgb_img)
    plt.title("Depth Map (RGB composite)")
    plt.axis('off')
    plt.show()

file_path = "../exported"

# file_mask = "depth_map_colour_"
file_mask = "depth_map_conf_"
# file_mask = "depth_map_grey_"
# file_mask = "depth_map_camera_"

file_list = [f for f in os.listdir(file_path) if f.startswith(file_mask) and f.endswith(".bin")]


# now iterate through the file_list
if not file_list:
    
    print("No files found matching the pattern.")

else:
    
    print("Files found:")
    # print the file list one name per line
    for file_name in file_list:
        print(file_name)

    for file_name in file_list:

        print(f"\nFile: {file_name}")

        # Load raw binary float32 data, little-endian
        data = np.fromfile(file_path + "/" + file_name, dtype='<f4')  # '<f4' = little-endian float32

        # get number of rows
        data_size = data.size
        num_rows = data.size // 6  

        print(f"Data size: {data_size}; Number of rows: {num_rows}")

    
        # print(f"Number of rows: {num_rows}")

        # Reshape the data into a 2D array with 6 columns
        data_reshaped = data.reshape(-1, 6)

        # print out the first 10 rows, neatly formatted with commas in five coloumns, the first column being the index
        num_rows_to_print = 10
        for i, row in enumerate(data_reshaped[:num_rows_to_print]):
            print(f"{i}, {row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}, {row[4]:.6f}, {row[5]:.6f}")   


        # Calculate mean of the last four columns and print 
        if num_rows > 0:
            mean_a = np.mean(data_reshaped[:, 2])
            mean_r = np.mean(data_reshaped[:, 3])
            mean_g = np.mean(data_reshaped[:, 4])
            mean_b = np.mean(data_reshaped[:, 5])
            # print file name, number of rows, mean a, r, g, b on the same line
            print(f"File: {file_name}, Number of rows: {num_rows}, Mean a: {mean_a:.6f}, Mean r: {mean_r:.6f}, Mean g: {mean_g:.6f}, Mean b: {mean_b:.6f}")
            
        # Show the depth map using matplotlib
        show_depth_map(data_reshaped)

