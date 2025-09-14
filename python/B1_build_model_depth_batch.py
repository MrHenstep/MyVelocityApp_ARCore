from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.dpt_depth import DPTDepthModel

import torch
import cv2
import numpy as np
import inspect
import re
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time

import L1_lib_extraction_and_visualisation as exv


class DepthAnythingWrapper(torch.nn.Module):
    """
    Accepts NumPy or Torch; shapes: HxWx3, 3xHxW, 1x3xHxW, 1xHxWx3.
    Values in [0,1] or [0,255]. Returns (1,1,h,w) on CPU, float32.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        """
        Initialize the depth estimation components.

        This constructor loads a pretrained image processor and depth estimation model
        identified by `model_id`, moves the model to the specified device, and switches
        it to evaluation mode.

        Args:
            model_id (str): Hugging Face model identifier or local path for the pretrained
                depth estimation model and its associated image processor.
            device (str, optional): Target device for model execution (e.g., "cpu", "cuda",
                "cuda:0"). Defaults to "cpu".

        Attributes:
            device (torch.device): Resolved PyTorch device used for inference.
            processor (transformers.AutoImageProcessor): Preprocessing pipeline bound to the model.
            model (transformers.AutoModelForDepthEstimation): Depth estimation model placed on
                the specified device and set to eval mode.

        Raises:
            OSError: If the processor or model cannot be loaded from the provided `model_id`.
        """
        super().__init__()
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_numpy_hwc3(x) -> np.ndarray:
        # Convert Torch → NumPy
        if isinstance(x, torch.Tensor):
            t = x.detach()
            # Remove batch if present
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            # 3xHxW → HxWx3
            if t.ndim == 3 and t.shape[0] == 3:
                t = t.permute(1, 2, 0)
            # 1xHxWx3 → HxWx3
            if t.ndim == 4 and t.shape[-1] == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            x = t.cpu().numpy()
        else:
            x = np.asarray(x)

        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"Expected an RGB image, got shape {x.shape}")

        x = x.astype(np.float32, copy=False)
        # Scale if it looks like 0–255
        if x.max() > 1.5:
            x = x / 255.0
        # Clamp for safety
        x = np.clip(x, 0.0, 1.0)
        return x

    @torch.no_grad()
    def forward(self, img) -> torch.Tensor:
        x = self._to_numpy_hwc3(img)  # HxWx3, float32 in [0,1]
        # inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False).to(self.device)
        out = self.model(**inputs)
        depth = getattr(out, "predicted_depth", getattr(out, "depth", None))
        if depth is None:
            raise RuntimeError("DepthAnythingWrapper: unexpected model outputs.")
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)  # (B,1,h,w)
        # Return CPU float32 for your downstream
        return depth.to("cpu", dtype=torch.float32)

    
def load_rgb_from_float6(file_path, file_name, H, W):
    """
    Loads RGB image data from a binary file containing float32 values in groups of six per pixel.
    The function reads the specified file, reshapes the data into rows of six floats per pixel,
    extracts the last three values (assumed to be RGB channels), and reshapes them into an image
    of shape (H, W, 3). If the RGB values are in the range [0, 255], they are normalized to [0, 1].
    Args:
        file_path (str): Path to the directory containing the file.
        file_name (str): Name of the binary file to read.
        H (int): Height of the output image.
        W (int): Width of the output image.
    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with float32 values in [0, 1].
    """

    data = np.fromfile(file_path + "/" + file_name, dtype='<f4').reshape(-1, 6)
    
    rgb = data[:, 3:6]                    # keep only r,g,b
    rgb = rgb.reshape(H, W, 3).astype(np.float32)

    # if values are already 0–1 floats, leave them
    # if they are 0–255 floats, divide by 255
    if rgb.max() > 1.5:
        rgb /= 255.0

    return rgb

def midas_pred_to_grey(pred, target_hw, mode="bilinear"):
    """
    Converts a MiDaS depth prediction to a normalized 3-channel grayscale image.
    Args:
        pred (np.ndarray or torch.Tensor): The depth prediction array or tensor. 
            Can be of shape (H, W), (C, H, W), (1, H, W), (B, C, H, W), etc.
        target_hw (tuple or list): Target (height, width) for resizing the output image.
        mode (str, optional): Interpolation mode for resizing. Default is "bilinear".
    Returns:
        np.ndarray: A 3-channel grayscale image of shape (H, W, 3), normalized to [0, 1].
    Raises:
        ValueError: If the input prediction shape is not supported.
    """
    H, W = map(int, target_hw)

    # to torch float tensor
    t = torch.as_tensor(pred).float()

    # ensure shape (1,1,h,w)
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        # (C,h,w) or (1,h,w)
        if t.shape[0] == 1:
            t = t.unsqueeze(0)              # -> (1,1,h,w)
        else:
            t = t[:1].unsqueeze(0)          # keep first channel
    elif t.ndim == 4:
        if t.shape[0] > 1:
            t = t[:1]                       # keep first batch
        if t.shape[1] > 1:
            t = t[:, :1]                    # keep first channel
    else:
        raise ValueError(f"Unexpected pred shape: {tuple(t.shape)}")

    # resize to (H, W)
    with torch.no_grad():
        out = F.interpolate(t, size=(H, W), mode=mode, align_corners=False)
        depth = out[0, 0].cpu().numpy().astype(np.float32)  # HxW

    # normalize to [0,1] for display
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-12:
        depth_grey = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_grey = (depth - dmin) / (dmax - dmin)

    # Convert to 3-channel grayscale (r=g=b)
    depth_rgb = np.stack([depth_grey]*3, axis=-1)  # shape (H, W, 3)

    return depth_rgb

def show_rgb_images(*images, titles=None, cmaps=None, vmaxs=None):
    """
    Display images in a grid with up to 4 columns per row.
    Args:
        *images: Images (e.g., rgb, depth_gray, etc.), each as np.ndarray.
        titles: Optional list of titles for each image.
        cmaps: Optional list of colormaps for each image (e.g., ["", "gray"]).
        vmaxs: Optional list of vmax values for each image (for normalization).
    """
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols

    if titles is None:
        frame = inspect.currentframe().f_back
        try:
            call_source = inspect.getframeinfo(frame).code_context[0]
            match = re.search(r'show_rgb_images\((.*?)\)', call_source)
            if match:
                args_str = match.group(1)
                args_str = args_str.split("titles=")[0].split("cmaps=")[0].split("vmaxs=")[0]
                names = [s.strip() for s in args_str.split(",")][:n]
                titles = names
            else:
                titles = [f"Image {i+1}" for i in range(n)]
        except Exception:
            titles = [f"Image {i+1}" for i in range(n)]
    if cmaps is None:
        cmaps = [None] * n
    if vmaxs is None:
        vmaxs = [None] * n

    plt.figure(figsize=(6 * cols, 6 * rows))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        plt.subplot(rows, cols, i + 1)
        disp = img
        if disp.dtype != np.float32 and disp.dtype != np.float64:
            disp = disp.astype(np.float32)
        if disp.max() > 1.5:
            disp = disp / 255.0
        cmap = cmaps[i] if i < len(cmaps) else None
        vmax = vmaxs[i] if i < len(vmaxs) else None
        if cmap is not None:
            plt.imshow(disp, cmap=cmap, vmin=0.0, vmax=vmax if vmax is not None else 1.0)
        else:
            plt.imshow(np.clip(disp, 0.0, 1.0))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def center_crop_square(img):
    """
    Center-crop the largest square fully contained in `img`.
    Returns:
      crop : np.ndarray of shape (S,S,...) where S = min(H,W)
      box  : (y0, y1, x0, x1) in original image coords
    """
    H, W = img.shape[:2]
    S = min(H, W)
    y0 = (H - S) // 2
    x0 = (W - S) // 2
    y1, x1 = y0 + S, x0 + S
    crop = img[y0:y1, x0:x1, ...] if img.ndim == 3 else img[y0:y1, x0:x1]
    return crop, (y0, y1, x0, x1)

def paste_center_square(square, full_hw, fill_value=0.0):
    """
    Paste a (256,256) square into the center of a blank canvas of size (H,W).

    Parameters
    ----------
    square : np.ndarray
        The cropped output (HxW or HxWxC), must be 256x256.
    full_hw : tuple
        (H, W) of the original image you want to paste into.
    fill_value : float
        Background fill value (default=0.0).

    Returns
    -------
    canvas : np.ndarray
        Canvas of shape (H,W) or (H,W,C) with the square inserted at the center.
    """
    H, W = full_hw
    h, w = square.shape[:2]
    if h != 256 or w != 256:
        raise ValueError(f"Expected square 256x256, got {h}x{w}")
    if H < h or W < w:
        raise ValueError("Full canvas must be larger than the square")

    # init blank canvas
    if square.ndim == 2:
        canvas = np.full((H, W), fill_value, dtype=square.dtype)
    else:
        C = square.shape[2]
        canvas = np.full((H, W, C), fill_value, dtype=square.dtype)

    # compute placement (centered)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    y1, x1 = y0 + h, x0 + w

    canvas[y0:y1, x0:x1, ...] = square
    return canvas

def paste_into_box(tile, full_hw, box, resize_if_needed=True, interp=cv2.INTER_LINEAR):
    """
    Paste `tile` (2D or 3D) into an H×W canvas at `box=(y0,y1,x0,x1)`.
    If tile size != box size and resize_if_needed=True, tile is resized to fit.
    """
    H, W = full_hw
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid box: {box}")
    if H < y1 or W < x1:
        raise ValueError(f"Box {box} exceeds canvas {H}×{W}")

    tile_arr = np.asarray(tile)
    if tile_arr.ndim == 2:
        canvas = np.zeros((H, W), dtype=tile_arr.dtype)
    else:
        C = tile_arr.shape[2]
        canvas = np.zeros((H, W, C), dtype=tile_arr.dtype)

    # Resize if needed to exactly match the box
    if (tile_arr.shape[0], tile_arr.shape[1]) != (h, w):
        if not resize_if_needed:
            raise ValueError(f"Tile {tile_arr.shape[:2]} != box {(h,w)} and resize_if_needed=False")
        # Handle 2D vs 3D resize
        if tile_arr.ndim == 2:
            tile_arr = cv2.resize(tile_arr, (w, h), interpolation=interp)
        else:
            tile_arr = cv2.resize(tile_arr, (w, h), interpolation=interp)

    canvas[y0:y1, x0:x1, ...] = tile_arr
    return canvas

def rot90k(arr, k: int):
    """
    Rotate a NumPy array by 90 degrees k times anti-clockwise.

    Parameters:
        arr (np.ndarray): The input array to rotate.
        k (int): Number of 90-degree rotations to apply (anti-clockwise). Only the remainder after dividing by 4 is used.

    Returns:
        np.ndarray: The rotated array.
    """
    k = int(k) % 4     # anti-clockwise steps
    if k == 0:
        return arr
    return np.rot90(arr, k=k)

def get_cropped_or_scaled_image(rgb, target_size, crop_not_scale):
    """
    Processes an input RGB image by either center-cropping it to a square and resizing,
    or directly resizing it to the target size.
    Args:
        rgb (np.ndarray): The input RGB image as a NumPy array.
        target_size (int): The desired width and height for the output image.
        crop_not_scale (bool): If True, center-crops the image to a square before resizing.
                               If False, directly resizes the image without cropping.
    Returns:
        input_image (np.ndarray): The processed image resized to (target_size, target_size).
        crop_box (tuple): The coordinates of the crop box as (y1, y2, x1, x2).
        height_in_image (int): The height of the image region used for resizing.
        width_in_image (int): The width of the image region used for resizing.
    """
    if crop_not_scale:
        cropped_image, crop_box = center_crop_square(rgb)
        input_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        height_in_image = cropped_image.shape[0]
        width_in_image = cropped_image.shape[1]
        # print(rgb.shape, "->", cropped_image.shape, "->", input_image.shape)
        # print("Crop box:", crop_box)       
    else:
        input_image = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        height_in_image = image_height
        width_in_image = image_width
        # print(rgb.shape, "->", input_image.shape)   
        crop_box = (0, image_height, 0, image_width)

    return input_image, crop_box, height_in_image, width_in_image

def get_prepared_image_for_model(input_image):
    """
    Prepares an input image for a deep learning model by normalizing its pixel values and converting it to a PyTorch tensor.
    The function normalizes the image using the ImageNet mean and standard deviation values, rearranges the image dimensions
    from (height, width, channels) to (channels, height, width), and adds a batch dimension.
    Args:
        input_image (np.ndarray): Input image as a NumPy array of shape (H, W, 3) with pixel values in [0, 1].
    Returns:
        torch.Tensor: Normalized image tensor of shape (1, 3, H, W) suitable for model input.
    """

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    input_image = (input_image - mean) / std

    input_image = torch.from_numpy(input_image.transpose(2,0,1)).unsqueeze(0)

    return input_image

def save_rgb_to_float6(file_path, img, alpha=1.0):
    """
    Saves an RGB image to a binary file as a float32 array with 6 channels per pixel: (x, y, alpha, R, G, B).
    Each pixel is represented by its x and y coordinates, an alpha value, and its RGB values.
    The output array has shape (H*W, 6), where H and W are the height and width of the input image.
    Parameters
    ----------
    file_path : str
        Path to the output binary file.
    img : np.ndarray
        Input image as a NumPy array of shape (H, W, 3), with RGB channels.
    alpha : float, optional
        Alpha value to assign to each pixel (default is 1.0).
    Returns
    -------
    np.ndarray
        The resulting (H*W, 6) float32 array containing [x, y, alpha, R, G, B] for each pixel.
    Raises
    ------
    ValueError
        If the input image does not have shape (H, W, 3).
    """

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {img.shape}")

    H, W, _ = img.shape

    # build coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(H, dtype=np.float32),
                                     np.arange(W, dtype=np.float32),
                                     indexing="ij")

    # flatten
    x_flat = x_coords.reshape(-1, 1)
    y_flat = y_coords.reshape(-1, 1)
    a_flat = np.full_like(x_flat, float(alpha), dtype=np.float32)
    rgb_flat = img.reshape(-1, 3).astype(np.float32)

    # concatenate into (N,6)
    arr = np.concatenate([x_flat, y_flat, a_flat, rgb_flat], axis=1)

    # write to file in float32
    arr.astype("<f4").tofile(file_path)

    return arr  # also return the array in case you want to use it

def resize_keep_aspect_upper_bound(img, long_side=384, mult=32):
    """
    Resizes an image while preserving its aspect ratio, ensuring the longer side matches the specified upper bound,
    and rounds the dimensions up to the nearest multiple of `mult`.

    Parameters:
        img (numpy.ndarray): Input image to be resized.
        long_side (int, optional): Desired length of the longer side after resizing. Default is 384.
        mult (int, optional): The multiple to which the new dimensions are rounded up. Default is 32.

    Returns:
        numpy.ndarray: The resized image with dimensions rounded to the nearest multiple of `mult`.
    """
    H, W = img.shape[:2]
    scale = long_side / max(H, W)
    newW, newH = int(round(W*scale)), int(round(H*scale))
    # round to nearest multiple of 32
    newW = max(mult, int(round(newW / mult)) * mult)
    newH = max(mult, int(round(newH / mult)) * mult)
    return cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)

def _patch_stochastic_depth_aliases_for_timm10(model):
    """
    Patches the stochastic depth aliases for models using the timm v0.10 API.

    This function checks if the given model's `pretrained.model` has a `blocks` attribute.
    For each block, if the block does not have a `drop_path` attribute, it assigns:
      - `drop_path1` to `drop_path` if available,
      - otherwise, assigns an `nn.Identity()` layer to `drop_path`.

    This ensures compatibility with code expecting a `drop_path` attribute in each block.

    Args:
        model: The model object containing a `pretrained.model` with `blocks`.

    Returns:
        None. The function modifies the model in place.
    """
    import torch.nn as nn
    blocks = getattr(model.pretrained.model, "blocks", None)
    if not blocks: return
    for blk in blocks:
        if not hasattr(blk, "drop_path"):
            if hasattr(blk, "drop_path1"):
                blk.drop_path = blk.drop_path1
            else:
                blk.drop_path = nn.Identity()

def load_model(model_name, model_weights_dict):
    """
    Loads a depth estimation model and its weights based on the specified model name.
    Parameters
    ----------
    model_name : str
        The name of the model to load. Supported values include:
        - "midas_v21_small"
        - "midas_v21"
        - "dpt_large"
        - "dpt_beit_large_512"
        - "depth_anything_v1_small"
        - "depth_anything_v1_base"
        - "depth_anything_v1_large"
        - "depth_anything_v2_small"
        - "depth_anything_v2_base"
        - "depth_anything_v2_large"
    model_weights_dict : dict
        A dictionary mapping model names to their corresponding weights file paths.
    Returns
    -------
    model : torch.nn.Module
        The loaded model instance.
    target_size : int or None
        The target input size for the model, or None if not applicable.
    Raises
    ------
    ValueError
        If an unknown model_name is provided.
    RuntimeError
        If an unknown ViT hidden size is encountered for "dpt_large".
    """
    weights_path = model_weights_dict.get(model_name)

    if model_name == "midas_v21_small":
        state_dict = torch.load(weights_path, map_location="cpu")
        model = MidasNet_small()
        target_size = 256
        model.load_state_dict(state_dict)

    elif model_name == "midas_v21":
        state_dict = torch.load(weights_path, map_location="cpu")
        model = MidasNet()
        target_size = 384
        model.load_state_dict(state_dict)

    elif model_name == "dpt_large":
        state_dict = torch.load(weights_path, map_location="cpu")
        hidden = state_dict["pretrained.model.cls_token"].shape[-1]
        backbone = "vitl16_384" if hidden == 1024 else ("vitb16_384" if hidden == 768 else None)
        if backbone is None:
            raise RuntimeError(f"Unknown ViT hidden size {hidden}; expected 768 or 1024.")
        model = DPTDepthModel(backbone=backbone, non_negative=True)
        model.load_state_dict(state_dict, strict=True)
        _patch_stochastic_depth_aliases_for_timm10(model)
        target_size = 384

    elif model_name == "dpt_beit_large_512":
        model = DPTDepthModel(backbone="beitl16_512", non_negative=True)
        state_dict = torch.load(weights_path, map_location="cpu")
        if any("relative_position_index" in k for k in state_dict):
            state_dict = {k: v for k, v in state_dict.items()
                          if "relative_position_index" not in k}
        _ = model.load_state_dict(state_dict, strict=False)  # filtered → non-strict
        _patch_stochastic_depth_aliases_for_timm10(model)
        target_size = 512

    elif model_name in (
        "depth_anything_v1_small", "depth_anything_v1_base", "depth_anything_v1_large",
        "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"
    ):

        model_id_map = {
            # V1 (LiheYoung org)
            "depth_anything_v1_small": "LiheYoung/depth-anything-small-hf",
            "depth_anything_v1_base":  "LiheYoung/depth-anything-base-hf",
            "depth_anything_v1_large": "LiheYoung/depth-anything-large-hf",

            # V2 (depth-anything org, Transformers-ready “-hf” variants)
            "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
            "depth_anything_v2_base":  "depth-anything/Depth-Anything-V2-Base-hf",
            "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
        }

        model = DepthAnythingWrapper(model_id=model_id_map[model_name], device="cpu")
        target_size = None

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, target_size

def get_model_input(image, model_name, crop_not_scale):
    """
    Prepares the input tensor and related metadata for various depth estimation models.
    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        model_name (str): The name of the model to prepare input for. Supported values include
            "midas_v21", "midas_v21_small", "dpt_large", "dpt_beit_large_512", and any string starting with "depth_anything_".
        crop_not_scale (bool): If True, crops the image instead of scaling (used for MiDaS models).
    Returns:
        tuple:
            tensor (np.ndarray): The processed image tensor suitable for the specified model.
            crop_box (tuple): The coordinates of the crop box (top, bottom, left, right) in the original image.
            height_in_image (int): The height of the region in the original image used for the model input.
            width_in_image (int): The width of the region in the original image used for the model input.
    Raises:
        ValueError: If an unknown model_name is provided.
    """
    global target_size  # ensure visible if you reuse it elsewhere

    if model_name in ("midas_v21", "midas_v21_small"):
        input_image, crop_box, height_in_image, width_in_image = \
            get_cropped_or_scaled_image(image, target_size, crop_not_scale)
        tensor = to_imagenet_norm(input_image)

    elif model_name == "dpt_large":
        input_image = resize_keep_aspect_upper_bound(image, 384, 32)
        crop_box = (0, image.shape[0], 0, image.shape[1])   # full frame
        height_in_image, width_in_image = image.shape[:2]
        tensor = to_minus1_1(input_image)

    elif model_name == "dpt_beit_large_512":
        input_image = resize_keep_aspect_upper_bound(image, 512, 32)
        crop_box = (0, image.shape[0], 0, image.shape[1])   # full frame
        height_in_image, width_in_image = image.shape[:2]
        tensor = to_minus1_1(input_image)

    elif model_name.startswith("depth_anything_"):
        input_image = image
        crop_box = (0, image.shape[0], 0, image.shape[1])
        height_in_image, width_in_image = image.shape[:2]
        tensor = input_image  # numpy; wrapper will process internally

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return tensor, crop_box, height_in_image, width_in_image

def to_imagenet_norm(input_image):
    """
    Normalizes an input image using ImageNet mean and standard deviation.

    Args:
        input_image (np.ndarray): Input image as a NumPy array of shape (H, W, C).
            The image can be in the range [0, 255] or [0, 1].

    Returns:
        torch.Tensor: Normalized image tensor of shape (1, C, H, W), suitable for PyTorch models.

    Notes:
        - The function automatically scales the image to [0, 1] if the maximum value is greater than 1.5.
        - The normalization uses the standard ImageNet mean ([0.485, 0.456, 0.406]) and
          standard deviation ([0.229, 0.224, 0.225]).
        - The output tensor is unsqueezed to add a batch dimension.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = input_image.astype(np.float32)
    if x.max() > 1.5: x /= 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)

def to_minus1_1(input_image):
    """
    Converts an input image to a PyTorch tensor with values normalized to the range [-1, 1].

    The function performs the following steps:
    1. Converts the input image to float32.
    2. If the maximum value in the image is greater than 1.5, scales the image by dividing by 255.
    3. Normalizes the image values from [0, 1] to [-1, 1] using the transformation: x = (x - 0.5) / 0.5.
    4. Transposes the image from (H, W, C) to (C, H, W) and adds a batch dimension.
    5. Converts the result to a PyTorch tensor.

    Args:
        input_image (np.ndarray): Input image array of shape (H, W, C), with values in [0, 255] or [0, 1].

    Returns:
        torch.Tensor: Normalized image tensor of shape (1, C, H, W) with values in [-1, 1].
    """
    x = input_image.astype(np.float32)
    if x.max() > 1.5: x /= 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)


if __name__ == "__main__":
        
    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # DATA - A
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
    # DATA - B
    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_30_2"
    # DATA - C
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_31_1"

    ###########################################################################################################
    
    CROP_NOT_SCALE = True

    # ROTATE_K = 3 # if the phone is vertical
    ROTATE_K = 0 # if the phone is horizontal

    MIDAS_MODEL_WEIGHTS = {
        "midas_v21_small": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\midas_v21_small-70d6b9c8.pt",
        "midas_v21": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\midas_v21-f6b98070.pt",
        "dpt_large": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\dpt_large-midas-2f21e586.pt",
        "dpt_beit_large_512": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\dpt_beit_large_512.pt"
        # Add more models and their weights paths here as needed
        
    }

    ##########################################################################################################

    # model_name_list = ["midas_v21", "dpt_large", "dpt_beit_large_512", "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"]
    model_name_list = ["depth_anything_v2_large"]

    ###############################################################################################################

    BATCH_NUMBER_LIST = [0, 1, 2, 3]
    # BATCH_NUMBER_LIST = [0]

    for batch_number in BATCH_NUMBER_LIST:

        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)
        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

        for index, row in enumerate(MATCHED_FILENAME_TABLE):

            # if (index != 10): continue

            file_name = row[1]  # camera file

            depth_grey_phone_original = load_rgb_from_float6(FILE_PATH, file_name.replace("camera", "grey"), 480, 640)
            depth_grey_phone = rot90k(depth_grey_phone_original, ROTATE_K)

            rgb_original = load_rgb_from_float6(FILE_PATH, file_name, 480, 640)
            rgb = rot90k(rgb_original, ROTATE_K)
            image_height, image_width = rgb.shape[:2]


            saved_images = []
            saved_images.append((file_name, rgb, None))
            saved_images.append((file_name.replace("camera", "grey"), depth_grey_phone, None))

            for model_name in model_name_list:
                
                print(f"Running {model_name} on image file {file_name} ...")

                print (f"\tLoading {model_name} ...")
                model, target_size = load_model(model_name, MIDAS_MODEL_WEIGHTS)

                print (f"\tPreparing input for {model_name} ...")
                input_image, crop_box, height_in_image, width_in_image = get_model_input(rgb, model_name, CROP_NOT_SCALE)

                print(f"\tRunning inference for {model_name} ...", end="", flush=True)
                start_time = time.time()
                
                with torch.no_grad():
                    pred = model(input_image)  
                
                end_time = time.time()
                print(f" {end_time - start_time:.2f} seconds")
                

                print (f"\tPost-processing output for {model_name} ...")
                depth_grey = midas_pred_to_grey(pred, (height_in_image, width_in_image), mode="bilinear")

                if model_name in ("midas_v21", "midas_v21_small"):
                    depth_on_full = paste_into_box(depth_grey, (image_height, image_width), crop_box, resize_if_needed=False)
                else:
                    depth_on_full = depth_grey  # already (H, W) == (image_height, image_width)

                
                depth_map_diff = -(depth_on_full - depth_grey_phone)/2.0 + 0.5

                print (f"\tSaving output for {model_name} ...")
                arr = save_rgb_to_float6(FILE_PATH + "/" + file_name.replace("camera", "MOD_"+model_name), rot90k(depth_on_full, -ROTATE_K), alpha=1.0)

                saved_images.append((model_name, depth_on_full, arr))

                print ("\n")

            show_rgb_images(*[img for _, img, _ in saved_images], titles=[name for name, _, _ in saved_images])