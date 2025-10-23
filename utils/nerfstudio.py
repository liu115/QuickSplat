from typing import List, Dict, Any, Tuple
import json

from pathlib import Path
import numpy as np
from .colmap import Camera, Image


def get_frame_names(json_path: Path) -> List[str]:
    with open(json_path, "r") as f:
        frame_data = json.load(f)

    frames = frame_data["frames"] + frame_data.get("test_frames", [])
    frame_names = [frame["file_path"] for frame in frames]
    return frame_names


def convert_camera(camera: Camera) -> Dict[str, Any]:
    camera_params = camera.params
    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": camera.width,
        "h": camera.height,
    }

    if camera.model == "OPENCV_FISHEYE":
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
                "camera_model": "OPENCV_FISHEYE",
            }
        )
    elif camera.model == "OPENCV":
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
                "camera_model": "OPENCV",
            }
        )
    else:
        # NOTE: Define PINHOLE with OPENCV model with all distortion parameters -> 0
        # Could replace with normal pinhole in the future
        out.update(
            {
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "camera_model": "OPENCV",
            }
        )
    return out


def convert_frames(images: Dict[int, Image]) -> List[Dict[str, Any]]:
    frames = []
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf
    for image_id, image in images.items():
        w2c = image.world_to_camera
        c2w = np.linalg.inv(w2c)

        # Convert from COLMAP's camera coordinate system to nerfstudio/instant-ngp
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        image_name = image.name.split("/")[-1]
        frame = {
            "file_path": image_name,
            "transform_matrix": c2w.tolist(),
        }
        min_x = min(min_x, c2w[0, 3])
        max_x = max(max_x, c2w[0, 3])
        min_y = min(min_y, c2w[1, 3])
        max_y = max(max_y, c2w[1, 3])
        min_z = min(min_z, c2w[2, 3])
        max_z = max(max_z, c2w[2, 3])

        frames.append(frame)
    # print(f"min_x: {min_x}, max_x: {max_x}")
    # print(f"min_y: {min_y}, max_y: {max_y}")
    # print(f"min_z: {min_z}, max_z: {max_z}")
    return frames
