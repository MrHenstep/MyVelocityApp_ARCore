import lib_extraction_and_visualisation as exv
import os

##########################################################################################################

FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_2_(test-static)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_2_(5ps)(test)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_1_(5fps)(car)(Ziggy)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_3_(test_rotation)"


##################################################################################################################

BATCH_NUMBER = 0
CONFIDENCE_LEVEL = 0.75
DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 25.0)

IMAGE_ORIENTATION_ROTATION = 0

DEPTH_POINTS_INDICES = range(0, 100, 1)

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

# exv.print_closest_ts_match(TIMESTAMPS_TABLE, MATCHED_INDICES)
MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, BATCH_NUMBER)

exv.batch_display_points_and_images(FILE_PATH, MATCHED_FILENAME_TABLE, CONFIDENCE_LEVEL, TIMESTAMPS_TABLE, MATCHED_INDICES, DEPTH_POINTS_INDICES, DEPTH_RANGE_FOR_COLOUR_MAP, rotation_deg=IMAGE_ORIENTATION_ROTATION)

for row in MATCHED_FILENAME_TABLE:
    # print(row)

    tracked_points = exv.read_float_data_as_nx4(FILE_PATH, row[5])
    print("Tracked points shape:", tracked_points.shape)
    print("Tracked points:", tracked_points)