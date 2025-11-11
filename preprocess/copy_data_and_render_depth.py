import argparse
import os
import sys
from pathlib import Path
import json

import imageio
import numpy as np
from tqdm import tqdm
try:
    import renderpy
except ImportError:
    print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
    sys.exit(1)

from submodules.scannetpp.common.utils.colmap import read_model
from submodules.scannetpp.common.scene_release import ScannetppScene_Release


def parse_args():
    parser = argparse.ArgumentParser(description="Render depth maps from scene files.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--scene_file', type=str, required=True, help='Path to the scene file.')
    parser.add_argument('--output_root', type=str, required=True, help='Directory to save the output data.')
    return parser.parse_args()


def render_depth_maps(
    data_root: str, output_root: str, scene_id: str,
):
    scene = ScannetppScene_Release(
        scene_id=scene_id,
        data_root=data_root,
    )

    transforms_path = Path(output_root) / "data" / scene_id / "transforms.json"
    with open(transforms_path, 'r') as f:
        transform = json.load(f)

    image_names = []
    # for frame in transform["frames"] + transform["test_frames"]:
    for frame in transform["test_frames"]:
        image_names.append(frame["file_path"].split("/")[-1])

    print(image_names)

    render_engine = renderpy.Render()
    render_engine.setupMesh(str(scene.scan_mesh_path))

    cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
    assert len(cameras) == 1, "Multiple cameras not supported"
    camera = next(iter(cameras.values()))

    # Render the original FISHEYE image
    # fx, fy, cx, cy = camera.params[:4]
    # params = camera.params[4:]
    # camera_model = camera.model

    # Render the undistorted image
    fx = transform["fl_x"]
    fy = transform["fl_y"]
    cx = transform["cx"]
    cy = transform["cy"]
    params = np.zeros(4, dtype=np.float32)  # No distortion parameters for undistorted images
    camera_model = "OPENCV"  # Use OpenCV model for undistorted
    render_engine.setupCamera(
        camera.height, camera.width,
        fx, fy, cx, cy,
        camera_model,
        params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
    )

    near = 0.05
    far = 20.0
    depth_dir = Path(output_root) / "data" / scene_id / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for image_id, image in tqdm(images.items(), "Rendering images"):
        if image.name not in image_names:
            continue
        world_to_camera = image.world_to_camera
        rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
        # Make depth in mm and clip to fit 16-bit image
        depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
        depth_name = image.name.split(".")[0] + ".png"
        imageio.imwrite(depth_dir / depth_name, depth)


def build_symlinks(
    data_root: str, output_root: str, scene_id: str,
):
    scene = ScannetppScene_Release(
        scene_id=scene_id,
        data_root=data_root,
    )

    dslr_undistorted_dir = scene.dslr_resized_undistorted_dir
    output_image_dir = Path(output_root) / "data" / scene_id / "images"
    # Create symlink for the undistorted images
    if not output_image_dir.exists():
        os.makedirs(output_image_dir.parent, exist_ok=True)
        os.symlink(dslr_undistorted_dir, output_image_dir)

    # Copy transforms.json
    output_transform_path = Path(output_root) / "data" / scene_id / "transforms.json"
    if not output_transform_path.exists():
        os.symlink(scene.dslr_nerfstudio_transform_undistorted_path, output_transform_path)
    else:
        print("Transform file already exists:", output_transform_path)


def main():
    args = parse_args()

    # Load the scene file
    with open(args.scene_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_root, exist_ok=True)

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        build_symlinks(
            data_root=args.data_root,
            output_root=args.output_root,
            scene_id=scene_id,
        )
        render_depth_maps(
            data_root=args.data_root,
            output_root=args.output_root,
            scene_id=scene_id,
        )


if __name__ == "__main__":
    main()
