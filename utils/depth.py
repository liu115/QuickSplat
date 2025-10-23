from typing import List
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


def depth_loss(depth_pred: torch.Tensor, depth_gt: torch.Tensor):
    assert depth_pred.shape == depth_gt.shape, f"Inconsistent shapes. depth_pred.shape: {depth_pred.shape}, depth_gt.shape: {depth_gt.shape}"
    mask = (depth_gt > 0)
    return F.l1_loss(depth_pred[mask], depth_gt[mask])


def log_depth_loss(depth_pred: torch.Tensor, depth_gt: torch.Tensor):
    assert depth_pred.shape == depth_gt.shape, f"Inconsistent shapes. depth_pred.shape: {depth_pred.shape}, depth_gt.shape: {depth_gt.shape}"
    mask = (depth_gt > 0)
    return F.l1_loss(torch.log10(depth_pred[mask] + 1e-6), torch.log10(depth_gt[mask] + 1e-6))


def depth_accuracy(depth_pred: torch.Tensor, depth_gt: torch.Tensor, threshold: float = 0.1):
    assert depth_pred.shape == depth_gt.shape, f"Inconsistent shapes. depth_pred.shape: {depth_pred.shape}, depth_gt.shape: {depth_gt.shape}"
    mask = (depth_gt > 0)
    diff = torch.abs(depth_pred[mask] - depth_gt[mask])
    accuracy = (diff < threshold).float().mean()
    return accuracy


def depth_rmse(depth_pred: torch.Tensor, depth_gt: torch.Tensor):
    assert depth_pred.shape == depth_gt.shape, f"Inconsistent shapes. depth_pred.shape: {depth_pred.shape}, depth_gt.shape: {depth_gt.shape}"
    mask = (depth_gt > 0)
    return torch.sqrt(F.mse_loss(depth_pred[mask], depth_gt[mask]))


def depth_delta(depth_pred: torch.Tensor, depth_gt: torch.Tensor):
    assert depth_pred.shape == depth_gt.shape, f"Inconsistent shapes. depth_pred.shape: {depth_pred.shape}, depth_gt.shape: {depth_gt.shape}"
    mask = (depth_gt > 0)
    depth_pred = depth_pred[mask]
    depth_gt = depth_gt[mask]

    ratio = torch.max(depth_pred / depth_gt, depth_gt / depth_pred)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < 1.25 ** 2).float().mean()
    delta3 = (ratio < 1.25 ** 3).float().mean()
    return delta1, delta2, delta3


def compute_full_depth_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    thresholds: List[float] = [0.02, 0.05, 0.1, 0.2],
):
    output_dict = {}
    output_dict["l1"] = depth_loss(depth_pred, depth_gt)
    output_dict["log_l1"] = log_depth_loss(depth_pred, depth_gt)

    for i, threshold in enumerate(thresholds):
        output_dict[f"accuracy_{threshold}"] = depth_accuracy(depth_pred, depth_gt, threshold)
    output_dict["rmse"] = depth_rmse(depth_pred, depth_gt)
    delta1, delta2, delta3 = depth_delta(depth_pred, depth_gt)
    output_dict["delta1"] = delta1
    output_dict["delta2"] = delta2
    output_dict["delta3"] = delta3
    return output_dict


def depth_to_color(
    depth: torch.Tensor,
    min_depth: float = 0.0,
    max_depth: float = 100.0,
) -> np.ndarray:
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = (depth * 255).astype(np.uint8)
    return depth


def save_depth(
    depth: torch.Tensor,
    save_path: str,
    min_depth: float = 0.0,
    max_depth: float = 100.0,
    downsample: int = 1,
):
    depth = depth_to_color(depth, min_depth, max_depth)
    depth = Image.fromarray(depth)
    if downsample > 1:
        depth = depth.resize((depth.width // downsample, depth.height // downsample), Image.NEAREST)
    depth.save(save_path)


def save_depth_opencv(
    depth_map,
    save_path,
    colormap=cv2.COLORMAP_MAGMA,
    normalize=True,
):
    """
    Converts a depth map to a colormap (heat map).

    Parameters:
        depth_map (numpy.ndarray): Grayscale depth map (2D array).
        colormap (int): OpenCV colormap (default is COLORMAP_JET).
        normalize (bool): Whether to normalize depth values to 0-255.

    Returns:
        numpy.ndarray: Colorized depth map (BGR image).
    """
    # COLORMAP_MAGMA, COLORMAP_INFERNO, COLORMAP_PLASMA, COLORMAP_JET

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.squeeze(0).cpu().numpy()

    if normalize:
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    depth_map = depth_map.astype(np.uint8)  # Convert to 8-bit image
    color_map = cv2.applyColorMap(depth_map, colormap)  # Apply colormap
    cv2.imwrite(str(save_path), color_map)
    return color_map
