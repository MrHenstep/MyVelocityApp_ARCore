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

if __name__ == "__main__":


    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_static_test"

    FILE_NAME_LIST = [
        "batch_0_trajectories_3D_Phone_midas_v21_small_Phone_openCV_LK.csv"
        ,"batch_0_trajectories_3D_Phone_midas_v21_small_CT2.csv"
        ,"batch_0_trajectories_3D_depth_anything_v2_large_Phone_openCV_LK.csv"
        ,"batch_0_trajectories_3D_depth_anything_v2_large_CT2.csv"
    ]

    BATCH_NUMBER_LIST = [0, 1, 2, 3]

    old_batch_mask = "batch_0"

    for batch_number in BATCH_NUMBER_LIST:

        batch_mask = f"batch_{batch_number}"

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

            trajectories[model_name] = trajectory_camera0     

            rms_abs_v = np.sqrt(np.mean(trajectory_camera0.loc[1:, "abs_v"]**2))
            print(f"RMS abs_v for {model_name}: {rms_abs_v:.4f}")

        plt.figure(figsize=(8, 5))
        color = '#4B6A88'  # Use the same blue-grey for all lines
        linestyles = ['-', '--', '-.', ':']
        for i, (model_name, traj_df) in enumerate(trajectories.items()):
            plt.plot(
            traj_df["t"],
            traj_df["abs_v"],
            label=model_name,
            color=color,
            linestyle=linestyles[i % len(linestyles)]
            )

        plt.xlabel("t (s)")
        plt.ylabel("|v| (ms$^{-1}$)")
        plt.title(f"Speed vs time, {batch_mask}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()