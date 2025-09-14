from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import L1_lib_extraction_and_visualisation as exv
import matplotlib.pyplot as plt


def read_trajectory_csv(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Parse a trajectory CSV like:
      <depth_model_name>, <tracker_model_name>
      Extrinsic Matrices:
      <4 lines per 4x4 matrix, blank line between matrices> × n
      Tracked points in camera frame:
      Frame k,<x>,<y>,<z>,1.0    × n (homogeneous, camera reference frame)
      Tracked points in reference frame:
      Frame k,<X>,<Y>,<Z>,1.0    × n (homogeneous, ARCore reference frame)

    Returns
    -------
    extrinsics_df : DataFrame
        index = frame (int), column 'matrix' = np.ndarray (4,4), float32
    cam_coords_df : DataFrame
        index = frame (int), column 'coord'  = np.ndarray (4,), homogeneous [x,y,z,1], float32
    ref_coords_df : DataFrame
        index = frame (int), column 'coord'  = np.ndarray (4,), homogeneous [X,Y,Z,1], float32
    meta : dict
        {'depth_model': ..., 'tracker_model': ...} if present on the first non-empty line.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    # metadata (model names on the first non-empty line)
    meta = {}
    first_nonempty = next((ln for ln in lines if ln), None)
    if first_nonempty:
        parts = [p.strip() for p in first_nonempty.split(",")]
        if len(parts) >= 2:
            meta["depth_model"] = parts[0]
            meta["tracker_model"] = parts[1]

    def find_line_idx(prefix: str) -> int:
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                return i
        return -1

    idx_ex  = find_line_idx("Extrinsic Matrices:")
    idx_cam = find_line_idx("Tracked points in camera frame:")
    idx_ref = find_line_idx("Tracked points in reference frame:")
    if idx_ex < 0 or idx_cam < 0 or idx_ref < 0:
        raise ValueError("File does not contain the expected section headers.")

    # parse 4×4 matrices, 4 lines per matrix
    def parse_matrices(start_idx: int, end_idx: int) -> Dict[int, np.ndarray]:
        mats, i, frame = {}, start_idx + 1, 0
        while i < end_idx:
            if not lines[i]:
                i += 1
                continue
            block = lines[i:i+4]
            if len(block) < 4:
                break
            rows, ok = [], True
            for r in block:
                vals = [v.strip() for v in r.split(",") if v.strip()]
                if len(vals) != 4:
                    ok = False; break
                try:
                    rows.append([float(v) for v in vals])
                except ValueError:
                    ok = False; break
            if not ok:
                break
            mats[frame] = np.array(rows, dtype=np.float32)
            frame += 1; i += 4
        return mats

    # parse "Frame k, x,y,z,w"
    def parse_coords(start_idx: int, end_idx: int) -> Dict[int, np.ndarray]:
        coords = {}
        for ln in lines[start_idx+1:end_idx]:
            if not ln or not ln.startswith("Frame "):
                continue
            try:
                after = ln[len("Frame "):]
                k_str, rest = after.split(",", 1)
                k = int(k_str.strip())
                vals = [float(x) for x in rest.split(",")]
                if len(vals) == 4:
                    coords[k] = np.array(vals, dtype=np.float32)
                elif len(vals) == 3:
                    coords[k] = np.array(vals + [1.0], dtype=np.float32)
            except Exception:
                continue
        return coords

    # section bounds
    end_ex, end_cam, end_ref = idx_cam, idx_ref, len(lines)

    ex_dict  = parse_matrices(idx_ex, end_ex)
    cam_dict = parse_coords(idx_cam, end_cam)
    ref_dict = parse_coords(idx_ref, end_ref)

    # build DataFrames keyed by frame
    extrinsics_df = pd.Series(ex_dict, name="matrix").to_frame().sort_index()
    cam_coords_df = pd.Series(cam_dict, name="coord").to_frame().sort_index()
    ref_coords_df = pd.Series(ref_dict, name="coord").to_frame().sort_index()

    # align to common frame indices (in case one section is shorter)
    common = extrinsics_df.index.intersection(cam_coords_df.index).intersection(ref_coords_df.index)
    if len(common) > 0:
        extrinsics_df = extrinsics_df.loc[common]
        cam_coords_df = cam_coords_df.loc[common]
        ref_coords_df = ref_coords_df.loc[common]

    return extrinsics_df, cam_coords_df, ref_coords_df, meta

def plot_trajectory_measures(
    trajectories, item, plot_label, title, y_min=None, y_max=None, shrink=1.0,
    measured_df=None, err_st_dev=1.0, last_frame=None, last_frame_diff_phone=0, save_filename=None
):
    """
    Plot trajectory measures with an optional shrink factor to scale the plot size and legend.
    If measured_df is provided, plot its non-null t_mid and |v| meas values as a series.
    For speed or dX, dY, dZ columns, plot against the average of the time in that row and the previous row.
    If save_filename is provided, save the plot as a PNG file.
    If item == "abs_r", return the final value minus the first value for each trajectory.
    """
    base_width, base_height = 8, 5
    plt.figure(figsize=(base_width * shrink, base_height * shrink))
    color_blue = '#0074D9'
    color_green = '#4E944F'
    linestyles = [':', '-', ':', '-']
    markers = ['o', 's', '^', 'D']
    skip_first = item in {"abs_v", "dX", "dY", "dZ"}

    abs_r_diffs = {}

    for i, (model_name, traj_df) in enumerate(trajectories.items()):
        marker = markers[i % len(markers)]
        color = color_green if i in (0, 1) else color_blue

        if item == "abs_R":
            # Calculate and print the difference between last and first value
            t_vals = traj_df["t"]
            y_vals = traj_df["abs_R"]
            if last_frame is not None:
                t_vals = t_vals[:last_frame + 1 + (last_frame_diff_phone if i < 2 else 0)]
                y_vals = y_vals[:last_frame + 1 + (last_frame_diff_phone if i < 2 else 0)]
            if len(traj_df) > 1 and "abs_R" in traj_df.columns:
                first_val = y_vals.iloc[0]
                last_val = y_vals.iloc[-1]
                diff = last_val - first_val
                abs_r_diffs[model_name] = diff
                print(f"{model_name}: Δ|R| = {diff:.4f}")
        elif skip_first:
            t_vals = traj_df["t"].values
            t_vals = (t_vals[1:] + t_vals[:-1]) / 2
            y_vals = traj_df[item].values[1:]
            if last_frame is not None:
                t_vals = t_vals[:last_frame + (last_frame_diff_phone if i < 2 else 0)]
                y_vals = y_vals[:last_frame + (last_frame_diff_phone if i < 2 else 0)]
        else:
            t_vals = traj_df["t"]
            y_vals = traj_df[item]
            if last_frame is not None:
                t_vals = t_vals[:last_frame + 1 + (last_frame_diff_phone if i < 2 else 0)]
                y_vals = y_vals[:last_frame + 1 + (last_frame_diff_phone if i < 2 else 0)]

        plt.plot(
            t_vals,
            y_vals,
            label=model_name,
            color=color,
            linestyle=linestyles[i % len(linestyles)],
            marker=marker,
            markerfacecolor='none'
        )

    if measured_df is not None and "t_mid" in measured_df.columns and "|v| meas" in measured_df.columns:
        mask = measured_df["t_mid"].notnull() & measured_df["|v| meas"].notnull()
        if mask.any():
            has_sig_v = "sig_v" in measured_df.columns
            has_sig_t = "sig_t" in measured_df.columns
            x = measured_df.loc[mask, "t_mid"]
            y = measured_df.loc[mask, "|v| meas"]
            xerr = err_st_dev * measured_df.loc[mask, "sig_t"] if has_sig_t else None
            yerr = err_st_dev * measured_df.loc[mask, "sig_v"] if has_sig_v else None
            plt.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt='o',
                label="Measured |v|",
                color='#A94442',
                markersize=1,
                linestyle='None',
                capsize=3
            )

    plt.xlabel("t (s)")
    plt.ylabel(plot_label)
    if title is not None:
        plt.title(f"{title}")
    plt.grid(True)
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
    plt.tight_layout()
    legend_fontsize = max(8, int(10 * shrink))
    plt.legend(fontsize=legend_fontsize)
    if save_filename is not None:
        plt.savefig(save_filename, format='png', dpi=150)
    plt.show()

    if item == "abs_R":
        print(abs_r_diffs)
        return abs_r_diffs

if __name__ == "__main__":


    ##########################################################################################################

    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # DATA - A
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
    # DATA - B
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
    # DATA - C
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

    ###########################################################################################################
    
    FILE_NAME_LIST = [
        "batch_0_trajectories_3D_Phone_midas_v21_small_Phone_openCV_LK.csv"
        ,"batch_0_trajectories_3D_Phone_midas_v21_small_CT2.csv"
        ,"batch_0_trajectories_3D_depth_anything_v2_large_Phone_openCV_LK.csv"
        ,"batch_0_trajectories_3D_depth_anything_v2_large_CT2.csv"
    ]

    BATCH_NUMBER_LIST = [0, 1, 2, 3]
    BATCH_NUMBER_LIST = [2]

    old_batch_mask = "batch_0"

    file_path_to_batch ={
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported":
            {0: (19, 0), 1: (19, 0), 2: (19, 0), 3: (19, 0)},
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test":
            {0: (19, 0), 1: (19, 0), 2: (19, 0), 3: (19, 0)},
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2":
            {0: (19, 0), 1: (17, 0), 2: (19, 0), 3: (19, 0)},
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1":
            {0: (15, -3), 1: (16, 0), 2: (19, 0), 3: (16, 0)},
    }

    file_path_to_label = {
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported":
            "X",
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test": 
            "A",    
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2":
            "B",
        "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1":
            "C",
    }

    plot_file_label = file_path_to_label.get(FILE_PATH, "X")
    plot_file_path = "C:\\Users\\steph\\OneDrive\\UCL\\T3 - MSc Project\\Report\\images\\4_results"



    for batch_number in BATCH_NUMBER_LIST:

        batch_to_last_frame = file_path_to_batch.get(FILE_PATH, {})

        batch_mask = f"batch_{batch_number}"

        measured_traj_path = Path(FILE_PATH) / f"batch_{batch_number}_measured_trajectory.csv"
        if measured_traj_path.exists():
            measured_df = pd.read_csv(measured_traj_path)
            print(f"Loaded measured trajectories: {measured_df.shape}")
        else:
            measured_df = None

        FILE_NAME_LIST = [fn.replace(old_batch_mask, batch_mask) for fn in FILE_NAME_LIST]
        old_batch_mask = batch_mask

        timestamps = exv.read_timestamp_files(FILE_PATH, batch_number)

        time_rebase = timestamps[0][2]

        trajectories = {}

        for file_name in FILE_NAME_LIST:

            model_name = file_name.replace(batch_mask + "_trajectories_3D_", "").replace(".csv", "")

            extrinsics_df, cam_coords_df, ref_coords_df, meta = read_trajectory_csv(FILE_PATH + "\\" + file_name)

            extrinsic_matrix_0 = extrinsics_df.loc[0, "matrix"]

            trajectory_camera0 = pd.DataFrame(columns=["t", "X", "Y", "Z", "dX", "dY", "dZ"], dtype=np.float32)

            for frame in extrinsics_df.index:
                
                extrinsic_matrix = extrinsics_df.loc[frame, "matrix"]
                combined_matrix = np.linalg.inv(extrinsic_matrix_0) @ extrinsic_matrix
                
                coords_cam_frame_0 = combined_matrix @ cam_coords_df.loc[frame, "coord"]
                X, Y, Z = coords_cam_frame_0[:3]

                if frame == 0:
                    X0, Y0, Z0 = X, Y, Z

                # print(f"Frame {frame}: Camera coords in frame 0: x={X:.4f}, y={Y:.4f}, z={Z:.4f}; Camera coords in frame 0 at frame 0: x0={X0:.4f}, y0={Y0:.4f}, z0={Z0:.4f}")

                abs_R = np.sqrt(X**2 + Y**2 + Z**2)
                abs_R_0 = np.sqrt((X-X0)**2 + (Y-Y0)**2 + (Z-Z0)**2)

                if frame > 0:
                    prev_coords = trajectory_camera0.loc[frame - 1, ["X", "Y", "Z"]].values
                    dX, dY, dZ = coords_cam_frame_0[:3] - prev_coords
                    dR = np.sqrt(dX**2 + dY**2 + dZ**2)
                    dt = timestamps[frame][2] - timestamps[frame - 1][2]
                    abs_v = dR / dt if dt > 0 else 0.0

                t = timestamps[frame][2] - time_rebase
                trajectory_camera0.loc[frame, "t"] = t
                trajectory_camera0.loc[frame, ["dX", "dY", "dZ", "abs_v"]] = (dX, dY, dZ, abs_v) if frame > 0 else (0.0, 0.0, 0.0, 0.0)
                trajectory_camera0.loc[frame, ["X", "Y", "Z"]] = coords_cam_frame_0[:3]
                trajectory_camera0.loc[frame, "abs_R"] = abs_R
                trajectory_camera0.loc[frame, "abs_R_0"] = abs_R_0

            trajectories[model_name] = trajectory_camera0     

            rms_abs_v = np.sqrt(np.mean(trajectory_camera0.loc[1:, "abs_v"]**2))
            print(f"RMS abs_v for {model_name}: {rms_abs_v:.4f}")

        last_frame = batch_to_last_frame.get(batch_number, None)

        _ = plot_trajectory_measures(trajectories, "X", "X (m)", None, shrink=0.7, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_X.png")
        _ = plot_trajectory_measures(trajectories, "Y", "Y (m)", None, shrink=0.7, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_Y.png")
        _ = plot_trajectory_measures(trajectories, "Z", "Z (m)", None, shrink=0.7, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_Z.png")

        _ = plot_trajectory_measures(trajectories, "abs_R", "|r| (m)", None, shrink=0.7, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_abs_R.png")

        _ = plot_trajectory_measures(trajectories, "abs_R_0", "|$r_{0}$| (m)", None, shrink=0.7, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_abs_R_0.png")

        # abs_r_drift_dict[batch_number] = abs_r_end

        # print (f"Final |r| - initial |r| for each model: {abs_r_end}")

        if batch_number == 1 or batch_number == 3:
            y_min = 0.0
            y_max = 1.0
        else:
            y_min = None
            y_max = None

        _ = plot_trajectory_measures(trajectories, "abs_v", "|v| (m/s)", None, shrink=0.7, measured_df=measured_df, err_st_dev=2.0, y_max=y_max, y_min=y_min, last_frame=last_frame[0], last_frame_diff_phone=last_frame[1], save_filename=Path(plot_file_path) / f"{plot_file_label}_{batch_mask}_abs_v.png")

        # Combine all trajectories into a single DataFrame with a model_name column
        combined_df = pd.concat(
            [df for model_name, df in trajectories.items()],
            keys=trajectories.keys(),
            names=["model_name", "frame"]
        ).reset_index()
        out_csv_path = Path(FILE_PATH) / f"{batch_mask}_all_trajectories_camera0.csv"
        combined_df.to_csv(out_csv_path, index=False)
        print(f"Saved all trajectories to {out_csv_path}")

