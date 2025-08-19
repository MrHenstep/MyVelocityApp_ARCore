import lib_extraction_and_visualisation as exv
import os

##########################################################################################################

FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_1_(frametiming)(indoors)(motion)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250812_2_(frametiming)(outdoors)(motion)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250813_1_(5fps)(outside)"
# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250818_1_(5fps)(outside)(tracking)"
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\20250819_1_(5fps)(car)(Ziggy)"

##################################################################################################################

BATCH_NUMBER = 0
CONFIDENCE_LEVEL = 0.75
DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 25.0)

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

exv.batch_display_points_and_images(FILE_PATH, MATCHED_FILENAME_TABLE, CONFIDENCE_LEVEL, TIMESTAMPS_TABLE, MATCHED_INDICES, DEPTH_POINTS_INDICES, DEPTH_RANGE_FOR_COLOUR_MAP)

for row in MATCHED_FILENAME_TABLE:
    # print(row)

    tracked_points = exv.read_float_data_as_nx4(FILE_PATH, row[5])
    print("Tracked points shape:", tracked_points.shape)
    print("Tracked points:", tracked_points)