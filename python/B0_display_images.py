import L1_lib_extraction_and_visualisation as exv
"""
B0_display_images.py
This module orchestrates batch visualization of depth points and corresponding images
exported by the Velociraptor Android application. It relies on helper functions from
`L1_lib_extraction_and_visualisation` to read timestamp metadata, match frames across
modalities, map matched indices to filenames, and render combined point/image displays.
High-level workflow:
1. Select a dataset export directory (FILE_PATH).
2. For each batch in BATCH_LIST:
    - Load per-modality timestamp tables.
    - Either compute closest timestamp matches across sources or enumerate all frames.
    - Map matched indices to filenames on disk.
    - Display images with overlaid depth points and optional rotation, filtering by confidence,
      and applying a depth-based color map.
Key configuration parameters:
- FILE_PATH (str):
  Absolute path to the exported dataset directory. Several example datasets are provided
  (comment/uncomment as needed). 
- CONFIDENCE_LEVEL (float):
  Minimum confidence threshold for depth points to be visualized. Points below this value
  may be filtered out during visualization.
- DEPTH_RANGE_FOR_COLOUR_MAP (tuple[float, float]):
  Depth range (min_depth_m, max_depth_m) in meters to map to the color scale used for
  the depth overlay. Values outside this range may be clipped or mapped to boundary colors.
- IMAGE_ORIENTATION_ROTATION (int):
  Rotation to apply to rendered images in degrees. Use 270 for phone-in-landscape
  acquisition, or 0 for portrait. Adjust as needed if the displayed orientation is incorrect.
- DEPTH_POINTS_INDICES (range or list[int]):
  Indices of depth points to visualize from the set extracted for each frame. For example,
  range(0, 20, 1) will render the first 20 points. Adjust to focus on subsets or all points.
- MATCH_TIMESTAMPS (bool):
  Controls how frames across modalities are paired:
     - True: Use closest timestamp matching (via `find_closest_timestamp_matches`) to align
        modalities with temporal tolerance and direction control.
     - False: Use `get_all_indices` to enumerate available frames without timestamp alignment.
- WEIGHTS_SIGMOID (tuple[float, float]):
  Parameters (X_CUT, X_WIDTH) defining a sigmoid weighting scheme that drops towards zero
  above X_CUT with a transition width X_WIDTH. This can be used by downstream visualization
  or fusion logic to de-emphasize points beyond a certain distance (e.g., > 7 m).
- BATCH_LIST (list[int]):
  List of batch identifiers to process. Each batch corresponds to a discrete set of captures
  (e.g., batch_0, batch_1). You can process multiple batches or limit to a single one for
  faster iteration.
External dependencies (provided by L1_lib_extraction_and_visualisation, imported as `exv`):
- read_timestamp_files(file_path: str, batch_number: int) -> dict-like
  Loads timestamp tables for each modality within the specified batch directory.
- find_closest_timestamp_matches(timestamps_table, max_gap_primary: int, max_gap_secondary: int, direction: str) -> list[tuple[int, ...]]
  Computes index correspondences across modalities by minimizing timestamp differences,
  with configurable search direction ('forward', 'backward', or 'both') and gap constraints.
- get_all_indices(file_path: str, batch_number: int) -> list[tuple[int, ...]] or list[int]
  Returns indices for frames available in the batch, without timestamp alignment.
- get_matched_filenames(matched_indices, file_path: str, batch_number: int) -> list[tuple[str, ...]]
  Maps matched indices to absolute file paths for the corresponding images and data files.
- batch_display_points_and_images(
     file_path: str,
     matched_filename_table,
     confidence_level: float,
     timestamps_table,
     matched_indices,
     depth_points_indices,
     depth_range_for_colour_map: tuple[float, float],
     rotation_deg: int = 0
  ) -> None
  Renders batched visualizations showing images with depth points overlaid, applying
  confidence filtering, depth color mapping, and rotation.
Usage:
- Set FILE_PATH to the desired export directory (uncomment the dataset you want).
- Adjust visualization parameters (confidence, depth range, rotation, indices).
- Set MATCH_TIMESTAMPS to True to align frames by timestamps, or False to iterate all frames.
- Set BATCH_LIST to the batch numbers you intend to process.
- Run the script to print progress per batch and open visualization windows or generate
  outputs as implemented by `batch_display_points_and_images`.
Notes:
- Paths are defined with escaped backslashes for Windows. Ensure the path exists and includes
  the expected batch subdirectories and timestamp files.
- If visualization windows do not appear or files are missing, verify that the export folder
  structure matches what `exv` expects and that the batch numbers are correct.
"""
import os

##########################################################################################################

# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

# DATA - A
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
# DATA - B
FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
# DATA - C
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

###########################################################################################################


CONFIDENCE_LEVEL = 0.75
DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, 5.0)

IMAGE_ORIENTATION_ROTATION = 270    # if the phone is horizontal
# IMAGE_ORIENTATION_ROTATION = 0      # if the phone is vertical

DEPTH_POINTS_INDICES = range(0, 20, 1)


MATCH_TIMESTAMPS = False

# Apply a sigmoid weight dropping towards zero above 7, width 1
X_CUT = 7.0
X_WIDTH = 1.0
WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

##################################################################################################################

BATCH_LIST = [0, 1, 2, 3]
BATCH_LIST = [3]


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
