from typing import Tuple, Dict, Literal, Optional
from pathlib import Path
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import open3d as o3d

from data.scannetpp import ScannetppDataset
from models.rasterizer import Camera, ScaffoldRasterizer
from models.rasterizer_2d import Scaffold2DGSRasterizer
from utils.utils import move_to_device


class MeshExtractor:

    def __init__(
        self,
        gaussian_type: Literal["2d", "3d"],
        voxel_size: float = 1.0,
        sdf_trunc: float = 0.04,
    ):
        self.gaussian_type = gaussian_type
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.bg_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")   # White

        if self.gaussian_type == "2d":
            self.rasterizer = Scaffold2DGSRasterizer(None, self.bg_color)
        elif self.gaussian_type == "3d":
            self.rasterizer = ScaffoldRasterizer(None, self.bg_color)
        else:
            raise ValueError(f"Unknown gaussian type {self.gaussian_type}")

    @torch.no_grad()
    def reconstruct_and_save(
        self,
        dataset: ScannetppDataset,
        gaussian_params: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        depth_source: Literal["median", "expected"] = "expected",
        depth_trunc: float = 3.0,
        crop_top: bool = True,
    ):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, x_cpu in enumerate(dataset):
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=x_cpu["width"],
                height=x_cpu["height"],
                fx=float(x_cpu["intrinsic"][0, 0]),
                fy=float(x_cpu["intrinsic"][1, 1]),
                cx=float(x_cpu["intrinsic"][0, 2]),
                cy=float(x_cpu["intrinsic"][1, 2]),
            )
            extrinsic = x_cpu["world_to_camera"].numpy()

            x_gpu = move_to_device(x_cpu, "cuda")
            camera = Camera(
                fov_x=x_cpu["fov_x"],
                fov_y=x_cpu["fov_y"],
                width=x_cpu["width"],
                height=x_cpu["height"],
                image=x_cpu["rgb"],
                world_view_transform=x_gpu["world_view_transform"],
                full_proj_transform=x_gpu["full_proj_transform"],
                camera_center=x_gpu["camera_center"],
            )

            render_outputs = self.rasterizer(gaussian_params, camera, active_sh_degree=0)

            if self.gaussian_type == "2d":
                depth = render_outputs[f"depth_{depth_source}"]

            elif self.gaussian_type == "3d":
                assert depth_source == "expected", "3D gaussian only supports expected depth"
                # Additionally render the depth
                depth = self.rasterizer.render_depth(gaussian_params, camera)

            image = (x_cpu["rgb"] * 255.0).permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype("uint8")
            depth = depth.squeeze(0).cpu().numpy()

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(image),
                o3d.geometry.Image(depth),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=extrinsic)

        mesh = volume.extract_triangle_mesh()
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
        faces = torch.from_numpy(np.asarray(mesh.triangles)).long()

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cleaned_mesh = self.clean_mesh(mesh)
            o3d.io.write_triangle_mesh(str(output_path).replace(".ply", "_raw.ply"), cleaned_mesh)
            if crop_top:
                xyz_min = np.min(np.asarray(mesh.vertices), axis=0)
                xyz_max = np.max(np.asarray(mesh.vertices), axis=0)
                xyz_max[2] = 2.5
                new_mesh = cleaned_mesh.crop(o3d.geometry.AxisAlignedBoundingBox(xyz_min[:, None], xyz_max[:, None]))
                # Crop out the top 15% of the mesh
                # new_xyz_max = (xyz_max - xyz_min) * np.array([1, 1, 0.85]) + xyz_min
                o3d.io.write_triangle_mesh(str(output_path), new_mesh)

            print(f"Saved mesh to {output_path}")
        return vertices, faces

    def clean_mesh(self, mesh, min_triangles=1000) -> o3d.geometry.TriangleMesh:
        # Copy the mesh to avoid modifying the original
        mesh = deepcopy(mesh)

        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()

        # Remove unconnected components
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles

        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh = mesh.remove_unreferenced_vertices()
        return mesh

