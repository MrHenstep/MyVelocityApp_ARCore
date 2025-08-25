import numpy as np
import lib_extraction_and_visualisation as exv


##################################################################################################################

def read_intrinsics_bin(file_name, image_dimensions_len=2, principal_point_len=2, focal_length_len=2):

    a = np.fromfile(file_name, dtype='<f4')  # little-endian float32
    i = 0
    
    image_dimensions  = a[i:i+image_dimensions_len]; i += image_dimensions_len
    principal_point   = a[i:i+principal_point_len];   i += principal_point_len
    focal_length      = a[i:i+focal_length_len]
    return image_dimensions.round().astype(np.int32), principal_point.astype(np.float32), focal_length.astype(np.float32)

def read_extrinsics_matrix(path):
    
    # ARCore Pose.toMatrix() writes a 4×4 matrix in column-major order.
    matrix = np.fromfile(path, dtype='<f4')  # little-endian float32

    if matrix.size != 16:
        raise ValueError(f"Expected 16 floats, got {matrix.size}")
    
    return matrix.reshape(4, 4, order='F')   # interpret column-major


if __name__ == "__main__":

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_1_(frametiming)(indoors)(motion)"
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_2_(frametiming)(outdoors)(motion)"
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250813_1_(5fps)(outside)"
    FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250818_1_(5fps)(outside)(tracking)"

    ###########################################################################################################

    ##################################################################################################################


    BATCH_NUMBER = 0
    CONFIDENCE_LEVEL = 0.75
    DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 25.0)

    DEPTH_POINTS_INDICES = range(0, 20, 19)

    MATCH_TIMESTAMPS = False

    # Apply a sigmoid weight dropping towards zero above 7, width 1
    X_CUT = 7.0
    X_WIDTH = 1.0
    WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

    ##################################################################################################################

    TIMESTAMPS_TABLE = exv.read_timestamp_files(FILE_PATH, BATCH_NUMBER)

    if MATCH_TIMESTAMPS:
        MATCHED_INDICES = exv.find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
    else:
        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, BATCH_NUMBER)

    MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, BATCH_NUMBER)


    ##################################################################################################################


    # for idx, row in enumerate(MATCHED_FILENAME_TABLE):

    #     FILE_NAME = row[7]
    #     image_dimensions, principal_point, focal_length = read_intrinsics_bin(FILE_PATH + "\\" +FILE_NAME)
    #     print(FILE_NAME + f" - Dim: {image_dimensions}, PP: {principal_point}, FL: {focal_length}")

    #     FILE_NAME = row[8]
    #     image_dimensions, principal_point, focal_length = read_intrinsics_bin(FILE_PATH + "\\" +FILE_NAME)
    #     print(FILE_NAME + f" - Dim: {image_dimensions}, PP: {principal_point}, FL: {focal_length}")

    #     FILE_NAME = row[6]
    #     extrinsic_matrix = read_extrinsics_matrix(FILE_PATH + "\\" + FILE_NAME)
    #     print(f"Extrinsic Matrix: {extrinsic_matrix}")

    #     print("\n")

    FILE_NAME = MATCHED_FILENAME_TABLE[0][0]

    pts = exv.read_float_data_as_nx4(FILE_PATH, FILE_NAME)

    print(f"Tracked points shape: {pts.shape}")
