import numpy as np


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

batch_number = 0 
nn = 0

FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\ARCore-velocity-app\\exported"

FILE_NAME =  f"batch_{batch_number}_camera_intrinsics_{nn}.bin"

image_dimensions, principal_point, focal_length = read_intrinsics_bin(FILE_PATH + "\\" +FILE_NAME)

print(f"Image Dimensions: {image_dimensions}")
print(f"Principal Point: {principal_point}")
print(f"Focal Length: {focal_length}")

FILE_NAME =  f"batch_{batch_number}_texture_intrinsics_{nn}.bin"

image_dimensions, principal_point, focal_length = read_intrinsics_bin(FILE_PATH + "\\" +FILE_NAME)

print(f"Image Dimensions: {image_dimensions}")
print(f"Principal Point: {principal_point}")
print(f"Focal Length: {focal_length}")

FILE_NAME =  f"batch_{batch_number}_extrinsic_matrix_{nn}.bin"

extrinsic_matrix = read_extrinsics_matrix(FILE_PATH + "\\" + FILE_NAME)

print(f"Extrinsic Matrix: {extrinsic_matrix}")
