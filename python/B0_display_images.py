import L1_lib_extraction_and_visualisation as exv
import os

##########################################################################################################

FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_static_test"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_28_1"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_28_4"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

##################################################################################################################


CONFIDENCE_LEVEL = 0.75
DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 5.0)

IMAGE_ORIENTATION_ROTATION = 270    # if the phone is horizontal
# IMAGE_ORIENTATION_ROTATION = 0      # if the phone is vertical

DEPTH_POINTS_INDICES = range(0, 20, 20)


MATCH_TIMESTAMPS = False

# Apply a sigmoid weight dropping towards zero above 7, width 1
X_CUT = 7.0
X_WIDTH = 1.0
WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

##################################################################################################################

BATCH_LIST = [0, 1, 2, 3]
BATCH_LIST = [1]


for batch_number in BATCH_LIST:

    print(f"Processing batch {batch_number}")

    TIMESTAMPS_TABLE = exv.read_timestamp_files(FILE_PATH, batch_number)


    if MATCH_TIMESTAMPS:
        MATCHED_INDICES = exv.find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
    else:
        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)

    # exv.print_closest_ts_match(TIMESTAMPS_TABLE, MATCHED_INDICES)
    MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

    exv.batch_display_points_and_images(FILE_PATH, MATCHED_FILENAME_TABLE, CONFIDENCE_LEVEL, TIMESTAMPS_TABLE, MATCHED_INDICES, DEPTH_POINTS_INDICES, DEPTH_RANGE_FOR_COLOUR_MAP, rotation_deg=IMAGE_ORIENTATION_ROTATION)
