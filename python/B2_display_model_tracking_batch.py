import os, glob
import time
import numpy as np
import torch
import L1_lib_extraction_and_visualisation as exv
import numpy as np
import cv2
import matplotlib.pyplot as plt



def read_float6(path):
    # returns (N,6): [x, y, alpha, r, g, b]
    return np.fromfile(path, dtype="<f4").reshape(-1, 6)

def float6_to_rgb(data, H, W):
    # indices in pixel space
    x = np.clip(np.round(data[:, 0]).astype(np.int32), 0, W-1)
    y = np.clip(np.round(data[:, 1]).astype(np.int32), 0, H-1)
    rgb = data[:, 3:6].astype(np.float32)

    # scale to 0..255 if values look like 0..1
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[y, x] = rgb  # if duplicates exist, later rows overwrite earlier ones
    return img  # H×W×3 uint8

def load_sequence_from_files(file_path, file_names, H, W):
    """
    file_paths: list[str] in the *temporal* order you want (frame 0..T-1)
    returns: frames_np shaped (T, H, W, 3) uint8
    """
    frames = []
    for p in file_names:
        data = read_float6(file_path + "\\" + p)
        frames.append(float6_to_rgb(data, H=H, W=W))
    return np.stack(frames, axis=0)

def to_video_tensor(frames_np, device="cpu"):
    # frames_np: (T, H, W, 3) uint8
    video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].float().to(device)  # 1×T×3×H×W
    return video

def make_queries(x0, y0, t0, device="cpu"):
    # shape 1×N×3 with rows [t, x, y]; here N=1
    return torch.tensor([[[float(t0), float(x0), float(y0)]]], dtype=torch.float32, device=device)

def render_tracks(frames_np, traj_xy, traj2_xy=None, vis=None, radius=3, vis_thresh=None):
    """
    frames_np is RGB and stays RGB for the caller.
    We convert to BGR only while calling OpenCV drawing, then back to RGB.
    """
    import numpy as np
    import cv2

    T, H, W, _ = frames_np.shape
    traj = traj_xy.copy()

    # Auto-rescale if values look normalized (0..1)
    if np.nanmax(traj) <= 2.0:
        traj[:, 0] *= W
        traj[:, 1] *= H
    if traj2_xy is not None:
        traj2 = traj2_xy.copy()
        if np.nanmax(traj2) <= 2.0:
            traj2[:, 0] *= W
            traj2[:, 1] *= H

    out = []
    for t in range(T):
        img_rgb = frames_np[t].copy()  # RGB
        x, y = traj[t]
        if traj2_xy is not None:
            x2, y2 = traj2[t]

        if not (np.isfinite(x) and np.isfinite(y)):
            out.append(img_rgb); continue

        if vis_thresh is not None and vis is not None and vis[t] < vis_thresh:
            out.append(img_rgb); continue

        xi, yi = int(round(x)), int(round(y))
        if traj2_xy is not None:
            x2, y2 = int(round(x2)), int(round(y2))

        if 0 <= xi < W and 0 <= yi < H:
            # Convert to BGR for OpenCV drawing
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # BGR colors for OpenCV
            color_bgr = (0, 255, 0)
            cv2.circle(img_bgr, (xi, yi), radius, color_bgr, -1)
            if traj2_xy is not None:
                cv2.circle(img_bgr, (x2, y2), radius*3, (255, 255, 255), 2)
            # Back to RGB for the rest of your pipeline
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        out.append(img_rgb)
    return np.stack(out, 0)  # RGB

def plot_frames_and_point(rotated, frame_indices=None):
    num_frames = len(rotated)
    num_frames_to_draw = len(frame_indices) if frame_indices is not None else num_frames

    cols = 4
    rows = int(np.ceil(num_frames_to_draw / cols))

    plt.figure(figsize=(16, rows * 4))

    frame_counter = 0
    for i in range(num_frames):

        if frame_indices is not None and i not in frame_indices:
            continue
        plt.subplot(rows, cols, frame_counter + 1)
        plt.imshow(rotated[i])          # already RGB; do NOT cvtColor here
        plt.axis("off")
        plt.title(f"Frame {i}")
        frame_counter += 1
    plt.tight_layout()
    plt.show()

def write_trajectories(file_path, batch_number, traj_xy):

    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
        raise ValueError(f"traj_xy must be shape (T,2); got {traj_xy.shape}")

    os.makedirs(file_path, exist_ok=True)

    for frame, (x, y) in enumerate(traj_xy):
        row = np.array([x, y, x, y], dtype="<f4")  # little-endian float32
        filename = f"batch_{batch_number}_tracked_point_MOD_CT2_{frame}.bin"
        out_path = os.path.join(file_path, filename)
        row.tofile(out_path)

    print(f"Wrote {traj_xy.shape[0]} files to {file_path}")

def read_trajectories(file_path, batch_number):
    """
    Loads traj_xy and vis from files written by write_trajectories.
    Returns:
        traj_xy: (T,2) numpy array of float32
        vis: (T,) numpy array of float32 (from the last column of each file)
    """
    pattern = os.path.join(file_path, f"batch_{batch_number}_tracked_point_MOD_CT2_*.bin")
    files = sorted(glob.glob(pattern), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    traj_xy = []
    vis = []
    for f in files:
        arr = np.fromfile(f, dtype="<f4")
        traj_xy.append(arr[:2])
        vis.append(arr[-1])
    traj_xy = np.stack(traj_xy, axis=0)
    vis = np.array(vis, dtype=np.float32)
    return traj_xy, vis

##################################################################################################

if __name__ == "__main__":

    # Load CoTracker 2
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(DEVICE).eval()

    IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640  

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # DATA - lot robot
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_28_2"
    

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

        INITIAL_POINT_FILE_NAME = f"batch_{batch_number}_tracked_point_0.bin"

        # ROTATE_FOR_DISPLAY = cv2.ROTATE_90_CLOCKWISE # if phone vertical
        ROTATE_FOR_DISPLAY = None # if phone horizontal
 

        # Compile frame file list
        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)
        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

        camera_filenames_list = [row[1] for row in MATCHED_FILENAME_TABLE]
        tracked_point_filenames_list = [row[5] for row in MATCHED_FILENAME_TABLE]

        tracked_point_coords_list = []
        for tracked_point_file in tracked_point_filenames_list:
            tracked_coords = exv.read_float_data_as_nxm(FILE_PATH, tracked_point_file)[0,:2]
            tracked_point_coords_list.append(tracked_coords)
        
        traj2_xy_np = np.stack(tracked_point_coords_list, axis=0)  # (T,2)

        frames = load_sequence_from_files(FILE_PATH, camera_filenames_list, H=480, W=640)     # (T,480,640,3)

        traj_xy, vis = read_trajectories(FILE_PATH, batch_number)

        painted = render_tracks(frames, traj_xy, traj2_xy=traj2_xy_np, vis=vis)

        if ROTATE_FOR_DISPLAY is not None:
            painted = np.array([cv2.rotate(img, ROTATE_FOR_DISPLAY) for img in painted])
        else:
            painted = painted  # leave as is

        # frame_indices = [7, 8, 9, 10, 11, 12, 13, 14]
        frame_indices = range(0, painted.shape[0])
        plot_frames_and_point(painted, frame_indices=frame_indices)

