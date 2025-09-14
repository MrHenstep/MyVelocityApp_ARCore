# def read_float_data_as_nx4(file_path, file_name, confidence_level=None):
import L1_lib_extraction_and_visualisation as exv
import numpy as np

##########################################################################################################

# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

# DATA - A
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
# DATA - B
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
# DATA - C
FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

###########################################################################################################

BATCH_NUMBER_LIST = [0, 1, 2, 3]
# BATCH_NUMBER_LIST = [0, 1]


for batch_number in BATCH_NUMBER_LIST:

    MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)
    MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

    consolidated_xy = []

    for iFrame, row in enumerate(MATCHED_FILENAME_TABLE):
        # print(row)

        phone_point_file_name = row[5]
        cotracker2_point_file_name = phone_point_file_name.replace("tracked_point", "tracked_point_MOD_CT2")

        phone_xy = exv.read_float_data_as_nxm(FILE_PATH, phone_point_file_name, m=4)[0, :2]
        cotracker2_xy = exv.read_float_data_as_nxm(FILE_PATH, cotracker2_point_file_name, m=5)[0, :2]  
        diff_xy = phone_xy - cotracker2_xy

        print(f"Frame {iFrame:3d} | Phone XY: {phone_xy[0]:8.3f}, {phone_xy[1]:8.3f} | "
            f"CoTracker2 XY: {cotracker2_xy[0]:8.3f}, {cotracker2_xy[1]:8.3f} | "
            f"Diff: {diff_xy[0]:8.3f}, {diff_xy[1]:8.3f}")
        
        consolidated_xy.append(np.hstack((phone_xy, cotracker2_xy, diff_xy)))

    consolidated_xy_arr = np.array(consolidated_xy)  # shape: (num_frames, 3, 2)

    # Compute mean, std, min, and max per column
    means = np.mean(consolidated_xy_arr, axis=0)
    stds = np.std(consolidated_xy_arr, axis=0)
    mins = np.min(consolidated_xy_arr, axis=0)
    maxs = np.max(consolidated_xy_arr, axis=0)

    stat_names = ["Mean", "Std", "Min", "Max"]
    stats = [means, stds, mins, maxs]

    sample_line = f"Stat {stat_names[0]:>4} | Phone XY: {0:8.3f}, {0:8.3f} | CoTracker2 XY: {0:8.3f}, {0:8.3f} | Diff: {0:8.3f}, {0:8.3f}"
    print("-" * len(sample_line))

    for stat_name, stat in zip(stat_names, stats):
        stat_str = " | ".join(f"{val:8.3f}" for val in stat)
        # print(f"{stat_name:>4}: {stat_str}")
        print(f"Stat {stat_name:>4} | Phone XY: {stat[0]:8.3f}, {stat[1]:8.3f} | "
        f"CoTracker2 XY: {stat[2]:8.3f}, {stat[3]:8.3f} | "
        f"Diff: {stat[4]:8.3f}, {stat[5]:8.3f}")
    print("\n")

    import pandas as pd
    trajectories_df = pd.DataFrame(consolidated_xy_arr, columns=["Phone_X", "Phone_Y", "CoTracker2_X", "CoTracker2_Y", "Diff_X", "Diff_Y"])

    print(trajectories_df)

    stem, root = MATCHED_FILENAME_TABLE[0][5].split("_0.")
    output_file_name = f"{stem}_trajectories.csv"


    trajectories_df.to_csv(FILE_PATH + "\\" + output_file_name, index=False)

