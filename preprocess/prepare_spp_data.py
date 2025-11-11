import os
import sys
import json
import argparse

import numpy as np
import open3d as o3d

from utils.colmap import read_model

sys.path.append("./submodules/scannetpp")
from common.scene_release import ScannetppScene_Release


def point3d_to_pcd(points3D):
    xyz, rgb = [], []
    for id, point in points3D.items():
        xyz.append(point.xyz)
        rgb.append(point.rgb)

    xyz = np.array(xyz)
    rgb = np.array(rgb) / 255.0

    # Remove nan or inf points
    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    rgb = rgb[mask]
    return xyz, rgb


def process_scene(data_root: str, output_root: str, scene_id: str, voxel_size: float):
    scene = ScannetppScene_Release(
        scene_id=scene_id,
        data_root=data_root,
    )

    mesh_path = scene.scan_mesh_path
    colmap_dir = scene.dslr_colmap_dir

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    # Save the mesh to output folder
    output_mesh_path = os.path.join(output_root, "mesh", f"{scene_id}.ply")
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)

    xyz_gt = np.asarray(mesh.vertices)
    xyz_min = np.min(xyz_gt, axis=0)
    xyz_max = np.max(xyz_gt, axis=0)
    print(f"Scene {scene_id} bbox: min {xyz_min}, max {xyz_max}")
    # Increase the bbox by 1.5x
    bbox_size = xyz_max - xyz_min
    center = (xyz_min + xyz_max) / 2.0
    xyz_min = center - (bbox_size * 0.75)
    xyz_max = center + (bbox_size * 0.75)
    print(f"Original size: {bbox_size}, new size: {xyz_max - xyz_min}")

    # Save the simplified mesh with normals for training (only the points) for training.
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pcd_gt_path = os.path.join(output_root, "mesh_simplified", f"{scene_id}.ply")
    os.makedirs(os.path.dirname(pcd_gt_path), exist_ok=True)
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = mesh.vertices
    pcd_downsampled.colors = mesh.vertex_colors
    pcd_downsampled.normals = mesh.vertex_normals
    pcd_downsampled = pcd_downsampled.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(pcd_gt_path, pcd_downsampled)

    # Save the colmap points (denoised to remove outliers)
    cameras, images, points3D = read_model(colmap_dir, ext=".txt")
    pcd_colmap_path = os.path.join(output_root, "colmap", f"{scene_id}.ply")
    os.makedirs(os.path.dirname(pcd_colmap_path), exist_ok=True)
    xyz, rgb = point3d_to_pcd(points3D)

    # Crop the points outside the bbox to avoid unnecessary large point clouds
    mask = np.all((xyz >= xyz_min) & (xyz <= xyz_max), axis=1)
    xyz = xyz[mask]
    rgb = rgb[mask]

    pcd_colmap = o3d.geometry.PointCloud()
    pcd_colmap.points = o3d.utility.Vector3dVector(xyz)
    pcd_colmap.colors = o3d.utility.Vector3dVector(rgb)

    pcd_colmap_new, ind = pcd_colmap.remove_radius_outlier(nb_points=5, radius=1.0)
    pcd_colmap_new = pcd_colmap_new.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud(pcd_colmap_path, pcd_colmap_new)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ScanNet++ SPP data for training"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the ScanNet++ released data root",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to the output prepared data root",
    )
    parser.add_argument(
        "--scene_file",
        type=str,
        required=True,
        help="Path to the scene list txt file",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling"
    )

    args = parser.parse_args()

    with open(args.scene_file, "r") as f:
        scene_ids = [line.strip() for line in f.readlines()]

    for scene_id in scene_ids:
        print(f"Processing scene {scene_id}...")
        process_scene(args.data_root, args.output_root, scene_id, args.voxel_size)
        print(f"Finished processing scene {scene_id}.")


if __name__ == "__main__":
    main()
