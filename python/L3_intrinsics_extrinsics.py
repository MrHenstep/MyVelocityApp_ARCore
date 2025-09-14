import numpy as np
import L1_lib_extraction_and_visualisation as exv


##################################################################################################################

def read_intrinsics_bin(file_name, image_dimensions_len=2, principal_point_len=2, focal_length_len=2):
    """
    Reads camera intrinsic parameters from a binary file.
    The binary file is expected to contain the following data in order:
    - Image dimensions (e.g., width and height), as float32 values.
    - Principal point coordinates (e.g., x and y), as float32 values.
    - Focal length values (e.g., fx and fy), as float32 values.
    Parameters
    ----------
    file_name : str
        Path to the binary file containing the intrinsic parameters.
    image_dimensions_len : int, optional
        Number of elements representing image dimensions (default is 2).
    principal_point_len : int, optional
        Number of elements representing the principal point (default is 2).
    focal_length_len : int, optional
        Number of elements representing the focal length (default is 2).
    Returns
    -------
    image_dimensions : np.ndarray
        Image dimensions as a rounded int32 numpy array.
    principal_point : np.ndarray
        Principal point coordinates as a float32 numpy array.
    focal_length : np.ndarray
        Focal length values as a float32 numpy array.
    """

    a = np.fromfile(file_name, dtype='<f4')  # little-endian float32
    i = 0
    
    image_dimensions  = a[i:i+image_dimensions_len]; i += image_dimensions_len
    principal_point   = a[i:i+principal_point_len];   i += principal_point_len
    focal_length      = a[i:i+focal_length_len]
    return image_dimensions.round().astype(np.int32), principal_point.astype(np.float32), focal_length.astype(np.float32)

def read_extrinsics_matrix(path):
    """
    Reads a 4x4 extrinsics matrix from a binary file written in column-major order.
    The file is expected to contain 16 little-endian float32 values, corresponding to a 4x4 matrix
    as written by ARCore's Pose.toMatrix() method. The matrix is reshaped to (4, 4) using column-major order.
    Args:
        path (str or Path): Path to the binary file containing the matrix.
    Returns:
        numpy.ndarray: A (4, 4) NumPy array representing the extrinsics matrix.
    Raises:
        ValueError: If the file does not contain exactly 16 float32 values.
    """
    
    # ARCore Pose.toMatrix() writes a 4×4 matrix in column-major order.
    matrix = np.fromfile(path, dtype='<f4')  # little-endian float32

    if matrix.size != 16:
        raise ValueError(f"Expected 16 floats, got {matrix.size}")
    
    return matrix.reshape(4, 4, order='F')   # interpret column-major


if __name__ == "__main__":

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"



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



    file_name = MATCHED_FILENAME_TABLE[0][0]

    pts = exv.read_float_data_as_nxm(FILE_PATH, file_name)

    print(f"Tracked points shape: {pts.shape}")
