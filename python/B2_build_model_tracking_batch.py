import os, glob
import time
import numpy as np
import torch
import L1_lib_extraction_and_visualisation as exv
import numpy as np
import cv2
import matplotlib.pyplot as plt



def read_float6(path):
    """
    Reads a binary file containing float32 data and reshapes it into an array of shape (N, 6).

    Each row in the returned array corresponds to a record with six float values:
    [x, y, alpha, r, g, b].

    Args:
        path (str): Path to the binary file to read.

    Returns:
        numpy.ndarray: Array of shape (N, 6) containing the parsed float values.

    Raises:
        ValueError: If the file size is not a multiple of 24 bytes (6 float32 values per record).
    """
    # returns (N,6): [x, y, alpha, r, g, b]
    return np.fromfile(path, dtype="<f4").reshape(-1, 6)

def float6_to_rgb(data, H, W):
    """
    Converts an array of float6 data points to an RGB image.
    Each row in `data` should contain six float values:
    - data[:, 0]: x-coordinate (pixel column)
    - data[:, 1]: y-coordinate (pixel row)
    - data[:, 3:6]: RGB color values (either in [0, 1] or [0, 255] range)
    The function maps each (x, y) position to its corresponding RGB value in an image of size (H, W).
    If RGB values are in [0, 1], they are scaled to [0, 255].
    If multiple data points map to the same pixel, later rows overwrite earlier ones.
    Args:
        data (np.ndarray): Array of shape (N, 6) containing float6 data points.
        H (int): Height of the output image.
        W (int): Width of the output image.
    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with dtype uint8.
    """
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
    Loads a sequence of frames from a list of files, converts each frame from float6 format to RGB, and stacks them into a NumPy array.
    Args:
        file_path (str): The directory path where the files are located.
        file_names (list of str): List of file names to load and process.
        H (int): Height of each frame.
        W (int): Width of each frame.
    Returns:
        np.ndarray: A NumPy array of shape (num_frames, H, W, 3) containing the stacked RGB frames.
    Note:
        This function assumes the existence of `read_float6` and `float6_to_rgb` functions for reading and converting the data.
    """

    frames = []
    for p in file_names:
        data = read_float6(file_path + "\\" + p)
        frames.append(float6_to_rgb(data, H=H, W=W))
    return np.stack(frames, axis=0)

def to_video_tensor(frames_np, device="cpu"):
    """
    Converts a NumPy array of video frames to a PyTorch tensor suitable for deep learning models.

    Args:
        frames_np (np.ndarray): A NumPy array of shape (T, H, W, 3) and dtype uint8,
            where T is the number of frames, H is the height, W is the width, and 3 is the number of color channels (RGB).
        device (str or torch.device, optional): The device on which to place the resulting tensor.
            Defaults to "cpu".

    Returns:
        torch.Tensor: A float tensor of shape (1, T, 3, H, W) on the specified device,
            representing the video batch with channels-first format.
    """
    # frames_np: (T, H, W, 3) uint8
    video = torch.tensor(frames_np).permute(0, 3, 1, 2)[None].float().to(device)  # 1×T×3×H×W
    return video

def make_queries(x0, y0, t0, device="cpu"):
    """
    Creates a PyTorch tensor representing a single query point with time, x, and y coordinates.

    Args:
        x0 (float): The x-coordinate of the query point.
        y0 (float): The y-coordinate of the query point.
        t0 (float): The time value associated with the query point.
        device (str or torch.device, optional): The device on which to create the tensor (default: "cpu").

    Returns:
        torch.Tensor: A tensor of shape (1, 1, 3) containing the query in the format [[t, x, y]].
    """
    # shape 1×N×3 with rows [t, x, y]; here N=1
    return torch.tensor([[[float(t0), float(x0), float(y0)]]], dtype=torch.float32, device=device)

def render_tracks(frames_np, traj_xy, vis=None, radius=3, vis_thresh=None):
    """
    Overlays trajectory points onto a sequence of RGB frames.
    Parameters
    ----------
    frames_np : np.ndarray
        Array of shape (T, H, W, 3) containing T RGB frames.
    traj_xy : np.ndarray
        Array of shape (T, 2) containing (x, y) coordinates for each frame.
        If coordinates are normalized (max <= 2.0), they are automatically rescaled to pixel values.
    vis : np.ndarray or None, optional
        Array of shape (T,) containing visibility scores for each frame. If provided, used to determine point color and optional thresholding.
    radius : int, optional
        Radius of the drawn trajectory points (default is 3).
    vis_thresh : float or None, optional
        If provided, points with visibility below this threshold are not drawn.
    Returns
    -------
    np.ndarray
        Array of shape (T, H, W, 3) containing RGB frames with trajectory points overlaid.
    Notes
    -----
    - Points are drawn in green if visibility >= 0.5, otherwise in red.
    - Frames with invalid coordinates or visibility below `vis_thresh` are returned unchanged.
    """
    import numpy as np
    import cv2

    T, H, W, _ = frames_np.shape
    traj = traj_xy.copy()

    # Auto-rescale if values look normalized (0..1)
    if np.nanmax(traj) <= 2.0:
        traj[:, 0] *= W
        traj[:, 1] *= H

    out = []
    for t in range(T):
        img_rgb = frames_np[t].copy()  # RGB
        x, y = traj[t]

        if not (np.isfinite(x) and np.isfinite(y)):
            out.append(img_rgb); continue

        if vis_thresh is not None and vis is not None and vis[t] < vis_thresh:
            out.append(img_rgb); continue

        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            # Convert to BGR for OpenCV drawing
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # BGR colors for OpenCV
            color_bgr = (0, 255, 0) if (vis is not None and vis[t] >= 0.5) else (0, 0, 255)
            cv2.circle(img_bgr, (xi, yi), radius, color_bgr, -1)
            # Back to RGB for the rest of your pipeline
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        out.append(img_rgb)
    return np.stack(out, 0)  # RGB

def plot_frames_and_point(rotated):
    """
    Plots a sequence of image frames in a grid layout.
    Args:
        rotated (list or array-like): A list or array of image frames (assumed to be in RGB format).
    Displays:
        A matplotlib figure showing each frame in a subplot, arranged in a grid with 5 columns.
        Each subplot is titled with its frame index, and axes are hidden for clarity.
    """
    num_frames = len(rotated)
    cols = 5
    rows = int(np.ceil(num_frames / cols))

    plt.figure(figsize=(16, rows * 4))
    for i in range(num_frames):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(rotated[i])          # already RGB; do NOT cvtColor here
        plt.axis("off")
        plt.title(f"Frame {i}")
    plt.tight_layout()
    plt.show()

def write_trajectories(file_path, batch_number, traj_xy, vis):
    """
    Writes trajectory data and visibility information for each frame to binary files.
    Each file contains a single row of five little-endian float32 values: [x, y, x, y, v],
    where (x, y) are the coordinates and v is the visibility for that frame.
    Parameters
    ----------
    file_path : str
        Directory path where the binary files will be saved. The directory will be created if it does not exist.
    batch_number : int
        Identifier for the batch, used in the output filenames.
    traj_xy : array-like of shape (T, 2)
        Sequence of (x, y) coordinates for T frames.
    vis : array-like of shape (T,)
        Sequence of visibility values corresponding to each frame.
    Raises
    ------
    ValueError
        If traj_xy does not have shape (T, 2) or if vis does not have the same length as traj_xy.
    Notes
    -----
    For each frame, a binary file named 'batch_{batch_number}_tracked_point_MOD_CT2_{frame}.bin'
    is created in the specified directory.
    """

    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    vis = np.asarray(vis, dtype=np.float32)
    if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
        raise ValueError(f"traj_xy must be shape (T,2); got {traj_xy.shape}")
    if vis.shape[0] != traj_xy.shape[0]:
        raise ValueError(f"vis must have same length as traj_xy; got {vis.shape[0]} vs {traj_xy.shape[0]}")

    os.makedirs(file_path, exist_ok=True)

    for frame, ((x, y), v) in enumerate(zip(traj_xy, vis)):
        row = np.array([x, y, x, y, v], dtype="<f4")  # little-endian float32
        filename = f"batch_{batch_number}_tracked_point_MOD_CT2_{frame}.bin"
        out_path = os.path.join(file_path, filename)
        row.tofile(out_path)

    print(f"Wrote {traj_xy.shape[0]} files to {file_path}")

##################################################################################################

if __name__ == "__main__":

    # Load CoTracker 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(DEVICE).eval()

    IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640  


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

        INITIAL_POINT_FILE_NAME = f"batch_{batch_number}_tracked_point_0.bin"

        # ROTATE_FOR_DISPLAY = cv2.ROTATE_90_CLOCKWISE # if phone vertical
        ROTATE_FOR_DISPLAY = None # if phone horizontal

        # Load initial point
        init_tracked_coords = exv.read_float_data_as_nxm(FILE_PATH, INITIAL_POINT_FILE_NAME)[0,:2]
        x0, y0 = init_tracked_coords
        t0 = 0
        queries = make_queries(x0, y0, t0, device=DEVICE)
        # print(x0, y0)


        # Compile frame file list
        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)
        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

        camera_filenames_list = [row[1] for row in MATCHED_FILENAME_TABLE]

        # Load frames
        frames = load_sequence_from_files(FILE_PATH, camera_filenames_list, H=480, W=640)     # (T,480,640,3)

        # Make video tensor
        video  = to_video_tensor(frames, device=DEVICE)            # (1,T,3,480,640)

        # Track points
        print("Tracking points...", flush=True)
        start_time = time.time()

        with torch.inference_mode():
            pred_tracks, pred_visibility = cotracker(video, queries=queries, grid_size=0)

        end_time = time.time()
        print(f" {end_time - start_time:.2f} seconds")

        # extract and visualise results
        traj_xy = pred_tracks[0, :, 0, :].detach().cpu().numpy()   # (T,2) pixel coords
        vis = pred_visibility[0, :, 0].detach().cpu().numpy()  # shape (T,)

        painted = render_tracks(frames, traj_xy, vis)
        
        if ROTATE_FOR_DISPLAY is not None:
            painted = np.array([cv2.rotate(img, ROTATE_FOR_DISPLAY) for img in painted])
        else:
            painted = painted  # leave as is

        plot_frames_and_point(painted)

        write_trajectories(
            file_path=FILE_PATH,
            batch_number=batch_number,
            traj_xy=traj_xy,
            vis=vis
        )



