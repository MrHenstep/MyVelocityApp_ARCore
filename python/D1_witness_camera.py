import cv2
import math
import matplotlib.pyplot as plt

def display_frames_between(
    video_path,
    start_time_s=0.0,
    end_time_s=None,
    assumed_fps=None,         # if None, use file's FPS; otherwise use this to map time->frames
    figsize=(12, 8),          # bigger figure size
    show_pts_ms=True,          # also show the file's reported PTS (POS_MSEC)
    skip_frames = 10
):
    """
    Displays selected frames from a video file between specified start and end times.
    Parameters:
        video_path (str): Path to the video file.
        start_time_s (float, optional): Start time in seconds. Defaults to 0.0.
        end_time_s (float or None, optional): End time in seconds. If None, displays until the end of the video. Defaults to None.
        assumed_fps (float or None, optional): Frames per second to use for time-to-frame mapping. If None, uses the video's FPS. Defaults to None.
        figsize (tuple, optional): Size of the matplotlib figure for displaying frames. Defaults to (12, 8).
        show_pts_ms (bool, optional): Whether to display the file-reported presentation timestamp (PTS) in milliseconds. Defaults to True.
        skip_frames (int, optional): Number of frames to skip between displays (i.e., display every Nth frame). Defaults to 10.
    Raises:
        IOError: If the video file cannot be opened.
        ValueError: If FPS is unknown or if end_time_s is less than start_time_s.
    Notes:
        - Frames are displayed using matplotlib.
        - The function attempts to use the video's FPS unless an assumed FPS is provided.
        - If the video's FPS or frame count cannot be determined, the function may raise an error or use fallback values.
        - The displayed frame title includes both the assumed timestamp and, optionally, the file-reported PTS.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    use_fps = assumed_fps if (assumed_fps and assumed_fps > 0) else (file_fps if file_fps > 0 else None)
    if use_fps is None:
        cap.release()
        raise ValueError("FPS is unknown. Provide `assumed_fps` explicitly or use a file with a readable FPS.")

    # Determine frame span from times (inclusive)
    start_frame = max(0, int(round(start_time_s * use_fps)))
    if end_time_s is None:
        # default to the end of the file if we know it; otherwise set a big number
        end_frame = (total_frames - 1) if total_frames is not None else 10**12
    else:
        end_frame = int(round(end_time_s * use_fps))
        if end_frame < start_frame:
            cap.release()
            raise ValueError("end_time_s must be >= start_time_s")

    # Clamp end_frame to file length if known
    if total_frames is not None:
        end_frame = min(end_frame, total_frames - 1)

    # Jump to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Iterate and display
    frame_idx = start_frame
    while frame_idx <= end_frame:


        ret, frame = cap.read()
        if not ret:
            break

        # Assumed timestamp from assumed/use_fps
        assumed_time_ms = (frame_idx / use_fps) * 1000.0

        if frame_idx % skip_frames == 0:

            # File-reported PTS in ms (may differ on VFR content)
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC) if show_pts_ms else None

            # Convert to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display bigger
            plt.figure(figsize=figsize)
            if show_pts_ms and pts_ms is not None and pts_ms > 0:
                plt.title(f"Frame {frame_idx} | assumed={assumed_time_ms:.3f} ms | PTS={pts_ms:.3f} ms")
            else:
                plt.title(f"Frame {frame_idx} | assumed={assumed_time_ms:.3f} ms")
            plt.imshow(frame_rgb)
            plt.axis("off")
            plt.show()

        frame_idx += 1

    cap.release()

# --- Example usage ---
# Show all frames between 2.5s and 7.0s, assuming 30 fps, with big figures.
# If you want to use the file's FPS instead, set assumed_fps=None.
# display_frames_between("your_video.mov", start_time_s=2.5, end_time_s=7.0, assumed_fps=30, figsize=(14, 9))

##########################################################################################################

# FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

# DATA - A
FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
# DATA - B
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
# DATA - C
# FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

###########################################################################################################

FILE_NAME = "batch_3_witness_camera.MOV"

start_time = 0.0
end_time = 10.0
skip_frames = 1

print("Num frames to include: ", (end_time - start_time) * 60 / skip_frames)



display_frames_between(FILE_PATH + "\\" + FILE_NAME, start_time_s=start_time, end_time_s=end_time, assumed_fps=60, figsize=(14, 9), skip_frames=skip_frames)