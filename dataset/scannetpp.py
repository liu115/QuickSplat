from typing import Optional, Dict, Any, List
import json
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import plyfile
from PIL import Image
from tqdm import tqdm
import cv2

from dataset.utils import (
    focal2fov,
    getWorld2View2,
    getProjectionMatrix,
    farthest_point_sampling,
    stack_data,
    get_nearest_camera_ids_mvs,
    apply_transform,
    apply_rotation,
    save_ply,
    get_translation_matrix,
    voxelize,
)
from dataset.augmentation import (
    random_perturb_np,
    random_dropout_np,
    random_add_np,
    random_global_3d_aug,
    random_crop_xy,
)


class ScannetppDataset(Dataset):
    def __init__(
        self,
        source_path: str,
        ply_path: str,
        scene_id: str,
        split: str,
        downsample: int = 1,
        num_train_frames: int = -1,
        subsample_randomness: bool = False,
        load_depth: bool = False,
    ):
        self.source_path = Path(source_path)
        self.ply_path = Path(ply_path)
        self.scene_id = scene_id
        self.split = split
        self.downsample = downsample
        self.load_depth = load_depth

        self.scene_path = self.source_path / scene_id / "images"
        self.depth_path = self.source_path / scene_id / "depth"
        self.json_path = self.source_path / scene_id / "transforms.json"
        self.num_train_frames = num_train_frames

        self.data = []
        with open(self.json_path, "r") as f:
            json_data = json.load(f)
            train_frames = json_data["frames"]
            train_frames.sort(key=lambda x: x["file_path"])
            test_frames = json_data["test_frames"]
            test_frames.sort(key=lambda x: x["file_path"])

            fx = json_data["fl_x"]
            fy = json_data["fl_y"]
            cx = json_data["cx"]
            cy = json_data["cy"]
            height = json_data["h"]
            width = json_data["w"]

        # Find out aabb:
        all_xyz = []
        for frame in train_frames + test_frames:
            # Convert the coordinate system to the same as COLMAP
            camera_to_world = np.array(frame["transform_matrix"])
            camera_to_world[2, :] *= -1
            camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
            camera_to_world[0:3, 1:3] *= -1
            all_xyz.append(camera_to_world[:3, 3])
        all_xyz = np.stack(all_xyz, axis=0)
        self.all_xyz = all_xyz

        if num_train_frames > 0 and len(train_frames) > num_train_frames:
            invalid_mask = np.ones(all_xyz.shape[0], dtype=bool)
            invalid_mask[:len(train_frames)] = False    # The test frames are invalid

            if subsample_randomness:
                generator = None
            else:
                generator = np.random.RandomState(0)

            sample_train_indices = farthest_point_sampling(
                all_xyz,
                num_train_frames,
                invalid_mask=invalid_mask,
                randomness=subsample_randomness,
                generator=generator,
            )
            train_frames = [train_frames[i] for i in sample_train_indices]

        self.num_train_frames = len(train_frames)
        self.num_test_frames = len(test_frames)

        if split == "train":
            all_frames = train_frames
        else:
            all_frames = test_frames
        self.fn_to_idx = {}

        for idx, frame in enumerate(all_frames):
            # Inverse mapping from file name to index
            self.fn_to_idx[frame["file_path"]] = idx

            image_path = self.scene_path / frame["file_path"]
            camera_to_world = np.array(frame["transform_matrix"])
            camera_to_world[2, :] *= -1
            camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
            camera_to_world[0:3, 1:3] *= -1

            world_to_camera = np.linalg.inv(camera_to_world)

            self.data.append({
                "file_path": image_path,
                "depth_path": self.depth_path / frame["file_path"].replace(".JPG", ".png"),
                "camera_to_world": camera_to_world,
                "world_to_camera": world_to_camera,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "height": height,
                "width": width,
            })

        self.scene_scale = self.get_global_scale(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self._load_frame(self.data, idx, bbox=None)
        frame["file_name"] = self.data[idx]["file_path"].name
        frame["sample_idx"] = idx
        frame["scene_id"] = self.scene_id
        return frame

    def get_aabb(self) -> torch.Tensor:
        camera_xyzs = torch.from_numpy(self.all_xyz).float()
        # Minus center * 1.5 + center
        camera_center = camera_xyzs.mean(dim=0)
        camera_xyzs -= camera_center
        camera_xyzs *= 1.5
        camera_xyzs += camera_center
        min_xyz = camera_xyzs.min(dim=0).values
        max_xyz = camera_xyzs.max(dim=0).values
        min_xyz[2] = 0
        max_xyz[2] = 5.0
        return torch.stack([min_xyz, max_xyz], dim=0)

    def get_global_scale(self, data) -> float:
        """Compute the distance between the center of all cameras to the farthest camera. Used in official gaussian splatting."""
        # nerf_normalization = getNerfppNorm(train_cam_infos)
        # scale = nerf_normalization["radius"]
        camera_centers = []
        for x in data:
            camera_centers.append(
                x["camera_to_world"][:3, 3]
            )
        camera_centers = np.array(camera_centers)   # N, 3
        avg_cam_center = np.mean(camera_centers, axis=0)
        dists = np.linalg.norm(camera_centers - avg_cam_center, axis=1)
        # translation = -avg_cam_center
        radius = np.max(dists) * 1.1
        return radius

    def load_points(self):
        ply_path = self.ply_path / f"{self.scene_id}.ply"
        with open(ply_path, "rb") as f:
            ply = plyfile.PlyData.read(f)
            vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
            color = np.vstack([ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]]).T
        vertices = vertices.astype(np.float32)
        color = color.astype(np.float32) / 255.0
        return vertices, color

    def load_mesh(self):
        ply_path = self.ply_path / f"{self.scene_id}.ply"
        with open(ply_path, "rb") as f:
            ply = plyfile.PlyData.read(f)
            vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
            color = np.vstack([ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]]).T
            faces = np.vstack(ply["face"].data["vertex_indices"])
        vertices = vertices.astype(np.float32)
        color = color.astype(np.float32) / 255.0
        faces = faces.astype(np.int32)
        return vertices, color, faces

    def _load_frame(self, frame_data, frame_idx: int, bbox: Optional[torch.Tensor]) -> Dict[str, Any]:
        x = frame_data[frame_idx]
        world_to_camera = x["world_to_camera"]
        image = Image.open(x["file_path"])
        width, height = image.size
        fx = x["fx"]
        fy = x["fy"]
        fov_x = focal2fov(fx, width)
        fov_y = focal2fov(fy, height)

        if self.downsample > 1:
            image = image.resize(
                (x["width"] // self.downsample, x["height"] // self.downsample),
                resample=Image.BILINEAR,
            )

        if self.load_depth and "depth_path" in x:
            if not os.path.exists(x["depth_path"]):
                raise FileNotFoundError(f"Depth path {x['depth_path']} does not exist")
            depth = cv2.imread(str(x["depth_path"]), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 1000.0
            if self.downsample > 1:
                depth = cv2.resize(
                    depth,
                    (width // self.downsample, height // self.downsample),
                    interpolation=cv2.INTER_NEAREST,
                )

        image = np.array(image, dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        fx = x["fx"] / self.downsample
        fy = x["fy"] / self.downsample
        cx = x["cx"] / self.downsample
        cy = x["cy"] / self.downsample
        fov_x = focal2fov(fx, width)
        fov_y = focal2fov(fy, height)
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)
        outputs = {
            "rgb": torch.from_numpy(image).permute(2, 0, 1).float(),
            "intrinsic": torch.from_numpy(intrinsic).float(),
            "camera_to_world": torch.from_numpy(x["camera_to_world"]).float(),
            "world_to_camera": torch.from_numpy(world_to_camera).float(),
            "fov_x": torch.tensor(fov_x),
            "fov_y": torch.tensor(fov_y),
            "height": torch.tensor(height),
            "width": torch.tensor(width),
        }
        if self.load_depth:
            outputs["depth"] = torch.from_numpy(depth)

        rot = np.transpose(world_to_camera[:3, :3])
        trans = world_to_camera[:3, 3]
        world_view_transform = torch.from_numpy(getWorld2View2(rot, trans)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=fov_x,
            fovY=fov_y,
        ).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        outputs["world_view_transform"] = world_view_transform
        outputs["projection_matrix"] = projection_matrix
        outputs["full_proj_transform"] = full_proj_transform
        outputs["camera_center"] = world_view_transform.inverse()[3, :3]

        if bbox is not None:
            outputs["bbox"] = bbox
        return outputs


class MultiScannetppDataset(ScannetppDataset):
    def __init__(
        self,
        source_path: str,
        ply_path: str,
        scene_list_path: str,
        num_samples: int,
        split: str,
        downsample: int = 1,
        num_train_frames: int = -1,
        load_depth: bool = False,
    ):
        self.source_path = Path(source_path)
        self.ply_path = Path(ply_path)
        self.scene_list_path = Path(scene_list_path)
        self.num_samples = num_samples
        self.split = split
        self.downsample = downsample
        self.num_train_frames = num_train_frames
        self.load_depth = load_depth

        self.scene_path = str(self.source_path / "{}/images")
        self.depth_path = str(self.source_path / "{}/depth")
        self.json_path = str(self.source_path / "{}/transforms.json")

        self.scene_list = []
        with open(self.scene_list_path, "r") as f:
            for line in f:
                self.scene_list.append(line.strip())
        self.scene_list.sort()

        self.data = []
        self.scene_id_to_idx = {}
        for i, scene_id in enumerate(tqdm(self.scene_list)):
            self.data.append(self.read_scene_data(scene_id))
            self.scene_id_to_idx[scene_id] = i

    def read_scene_data(self, scene_id) -> List[Dict[str, Any]]:
        data = []

        with open(self.json_path.format(scene_id), "r") as f:
            json_data = json.load(f)
            train_frames = json_data["frames"]
            train_frames.sort(key=lambda x: x["file_path"])
            test_frames = json_data["test_frames"]
            test_frames.sort(key=lambda x: x["file_path"])

            fx = json_data["fl_x"]
            fy = json_data["fl_y"]
            cx = json_data["cx"]
            cy = json_data["cy"]
            height = json_data["h"]
            width = json_data["w"]

        # Find out aabb:
        all_xyz = []
        for frame in train_frames + test_frames:
            camera_to_world = np.array(frame["transform_matrix"])
            camera_to_world[2, :] *= -1
            camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
            camera_to_world[0:3, 1:3] *= -1
            all_xyz.append(camera_to_world[:3, 3])
        all_xyz = np.stack(all_xyz, axis=0)

        if self.num_train_frames > 0 and len(train_frames) > self.num_train_frames:
            invalid_mask = np.ones(all_xyz.shape[0], dtype=bool)
            invalid_mask[:len(train_frames)] = False
            sample_train_indices = farthest_point_sampling(
                all_xyz,
                self.num_train_frames,
                invalid_mask=invalid_mask,
                randomness=False,
            )
            train_frames = [train_frames[i] for i in sample_train_indices]

        if self.split == "train":
            all_frames = train_frames
        elif self.split == "all":
            all_frames = train_frames + test_frames
        else:
            all_frames = test_frames

        for idx, frame in enumerate(all_frames):
            # Inverse mapping from file name to index
            # self.fn_to_idx[frame["file_path"]] = idx

            image_path = Path(self.scene_path.format(scene_id)) / frame["file_path"]
            depth_path = Path(self.depth_path.format(scene_id)) / frame["file_path"].replace(".JPG", ".png")
            # Convert the coordinate system to the same as COLMAP
            camera_to_world = np.array(frame["transform_matrix"])
            camera_to_world[2, :] *= -1
            camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
            camera_to_world[0:3, 1:3] *= -1

            world_to_camera = np.linalg.inv(camera_to_world)

            data.append({
                "file_path": image_path,
                "depth_path": depth_path,
                "camera_to_world": camera_to_world,
                "world_to_camera": world_to_camera,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "height": height,
                "width": width,
            })

        return data

    def load_points(self, scene_id):
        ply_path = self.ply_path / f"{scene_id}.ply"
        with open(ply_path, "rb") as f:
            ply = plyfile.PlyData.read(f)
            vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
            color = np.vstack([ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]]).T
        vertices = vertices.astype(np.float32)
        color = color.astype(np.float32) / 255.0
        return vertices, color

    def load_points_with_normal(self, scene_id):
        ply_path = self.ply_path / f"{scene_id}.ply"
        with open(ply_path, "rb") as f:
            ply = plyfile.PlyData.read(f)
            vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
            color = np.vstack([ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]]).T
            normal = np.vstack([ply["vertex"]["nx"], ply["vertex"]["ny"], ply["vertex"]["nz"]]).T
        vertices = vertices.astype(np.float32)
        color = color.astype(np.float32) / 255.0
        normal = normal.astype(np.float32)
        return vertices, color, normal

    def get_global_scale(self, scene_id: str) -> float:
        scene_data = self.data[self.scene_id_to_idx[scene_id]]
        return super().get_global_scale(scene_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, scene_idx: int) -> Dict[str, Any]:
        # Get all the frames from the scene and sample num_samples frames using farthest point sampling
        scene_data = self.data[scene_idx]
        scene_id = self.scene_list[scene_idx]
        num_frames = len(scene_data)
        # print(f"Loading {num_frames} frames from scene {scene_id} (idx: {scene_idx})")

        scene_cam_xyzs = []
        for x in scene_data:
            scene_cam_xyzs.append(x["camera_to_world"][:3, 3])
        scene_cam_xyzs = np.stack(scene_cam_xyzs, axis=0)

        invalid_mask = np.zeros(num_frames, dtype=bool)
        sample_indices = farthest_point_sampling(
            scene_cam_xyzs,
            self.num_samples,
            invalid_mask=invalid_mask,
            randomness=True,
        )

        frames = []
        for idx in sample_indices:
            frame_data = self._load_frame(scene_data, idx, bbox=None)
            frame_data["scene_id"] = scene_id
            frame_data["sample_idx"] = idx
            frame_data["file_name"] = scene_data[idx]["file_path"].name
            frames.append(frame_data)

        frames = stack_data(frames)
        return frames


class MultiScannetppPointDataset(ScannetppDataset):
    def __init__(
        self,
        source_path: str,
        ply_path: str,
        gt_ply_path: str,
        scene_list_path: str,
        max_points: int = 600_000,
        voxel_size: float = 0.02,
        global_aug: bool = False,
        noise_aug: bool = False,
        color_aug: bool = False,
        voxelize_aug: bool = False,
        read_normal: bool = False,
        return_gt: bool = False,
        read_normal_gt: bool = False,
    ):
        self.source_path = Path(source_path)
        self.ply_path = Path(ply_path)
        self.scene_list_path = Path(scene_list_path)

        self.scene_list = []
        with open(self.scene_list_path, "r") as f:
            for line in f:
                self.scene_list.append(line.strip())
        self.scene_list.sort()

        self.data = []
        self.scene_id_to_idx = {}
        for i, scene_id in enumerate(tqdm(self.scene_list)):
            # self.data.append(self.read_scene_data(scene_id))
            self.scene_id_to_idx[scene_id] = i

        self.max_points = max_points
        self.voxel_size = voxel_size

        # Careful with this one. It will change the point cloud coordinates (affects the camera poses)
        # Need to inverse the transform before rendering to camera poses
        self.global_aug = global_aug
        self.noise_aug = noise_aug
        self.color_aug = color_aug
        self.voxelize_aug = voxelize_aug
        self.read_normal = read_normal

        # For GT points from the mesh
        self.gt_ply_path = Path(gt_ply_path)
        self.return_gt = return_gt
        self.read_normal_gt = read_normal_gt

    def load_ply(self, ply_path: Path, read_normal: bool = False):
        with open(ply_path, "rb") as f:
            ply = plyfile.PlyData.read(f)
            vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T
            color = np.vstack([ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]]).T

            if read_normal:
                assert "nx" in ply["vertex"], "Normal not found in the ply file"
                normals = np.vstack([ply["vertex"]["nx"], ply["vertex"]["ny"], ply["vertex"]["nz"]]).T
        vertices = vertices.astype(np.float32)
        color = color.astype(np.float32) / 255.0

        if read_normal:
            normals = normals.astype(np.float64)
            return vertices, color, normals

        return vertices, color

    def __getitem__(self, scene_idx):
        scene_id = self.scene_list[scene_idx]

        # Load the point cloud
        ply_path = self.ply_path / f"{scene_id}.ply"
        if self.read_normal:
            xyz, rgb, normal = self.load_ply(ply_path, read_normal=True)
        else:
            xyz, rgb = self.load_ply(ply_path)

        # Color augmentation
        if self.color_aug and np.random.rand() < 0.5:
            # Randomly adjust the color of the source point cloud
            rgb = rgb + np.random.uniform(-0.1, 0.1, size=rgb.shape).astype(np.float32)
            rgb = np.clip(rgb, 0, 1)

        if self.noise_aug and np.random.rand() < 0.5:
            xyz, rgb = random_perturb_np(xyz, rgb, noise_level=self.voxel_size, clip=self.voxel_size * 2)
        if self.noise_aug and np.random.rand() < 0.5:
            xyz, rgb = random_dropout_np(xyz, rgb, dropout_ratio_range=(0.1, 0.25))

        xyz_world = xyz.copy()      # Preserve the original point cloud before global augmentation and voxelization

        if self.return_gt:
            # Read the GT point cloud to get the bounding box
            gt_ply_path = self.gt_ply_path / f"{scene_id}.ply"
            if self.read_normal_gt:
                xyz_gt, rgb_gt, normal_gt = self.load_ply(gt_ply_path, read_normal=True)
                normal_gt = normal_gt / np.linalg.norm(normal_gt, axis=1)[:, None].clip(min=1e-9)
            else:
                xyz_gt, rgb_gt = self.load_ply(gt_ply_path)
            xyz_world_gt = xyz_gt.copy()

        # Global 3D augmentation: random flip, scale, rotate, and shift
        if self.global_aug:
            transform = random_global_3d_aug(rot90_only=False)
            xyz = apply_transform(xyz, transform)
            if self.crop_points or self.return_gt:
                xyz_gt = apply_transform(xyz_gt, transform)
                if self.read_normal_gt:
                    normal_gt = apply_rotation(normal_gt, transform[:3, :3])
        else:
            transform = np.eye(4)

        # Random shuffle (in case the voxelization always remove the same duplicates)
        if self.voxelize_aug:
            indices = np.random.permutation(len(xyz))
        else:
            indices = np.arange(len(xyz))

        xyz_world = xyz_world[indices]
        xyz = xyz[indices]
        rgb = rgb[indices]
        if self.read_normal:
            normal = normal[indices]
            xyz_world = np.concatenate([xyz_world, normal], axis=1)

        bbox_min = np.min(xyz, axis=0)
        bbox_max = np.max(xyz, axis=0)

        if self.voxelize_aug:
            # Randomly move the points before voxelization
            # We also want to move the points a bit away from the origin to avoid fitting to the boundary
            OFFSET_SCALE = 32
            bbox_min -= np.random.uniform(0, self.voxel_size * OFFSET_SCALE, size=3)

        # print(f"Before voxelization: bbox_min={bbox_min}, bbox_max={bbox_max}")
        (
            _xyz,
            rgb,
            xyz_voxel,
            xyz_offset,
            bbox,
            voxelize_tranform,
            xyz_world,
        ) = voxelize(xyz, rgb, self.voxel_size, bbox_min=bbox_min, bbox_max=bbox_max, xyz_world=xyz_world)
        if self.return_gt:

            if self.read_normal_gt:
                # So that both xyz_world_gt and normal_gt won't be effected by the voxelization transform (scaling + translation)
                xyz_world_gt = np.concatenate([xyz_world_gt, normal_gt], axis=1)
            (
                _xyz_gt,
                rgb_gt,
                xyz_voxel_gt,
                xyz_offset_gt,
                bbox_gt,
                voxelize_tranform_gt,
                xyz_world_gt,
            ) = voxelize(xyz_gt, rgb_gt, self.voxel_size, bbox_min=bbox[0], bbox_max=bbox[1], xyz_world=xyz_world_gt)
            if self.read_normal_gt:
                normal_gt = xyz_world_gt[:, 3:]
                xyz_world_gt = xyz_world_gt[:, :3]

        if self.read_normal:
            normal = xyz_world[:, 3:]
            xyz_world = xyz_world[:, :3]

        # Subsample the point cloud if needed
        if xyz_voxel.shape[0] > self.max_points:
            indices = np.random.choice(xyz_voxel.shape[0], self.max_points, replace=False)
            xyz_voxel = xyz_voxel[indices]
            xyz_offset = xyz_offset[indices]
            rgb = rgb[indices]
            xyz_world = xyz_world[indices]
            if self.read_normal:
                normal = normal[indices]

        transform = voxelize_tranform @ transform   # From world to voxel
        transform = np.linalg.inv(transform)    # From voxel to world

        bbox_voxel_min = np.min(xyz_voxel, axis=0)
        bbox_voxel_max = np.max(xyz_voxel, axis=0)
        bbox_voxel = np.stack([bbox_voxel_min, bbox_voxel_max], axis=0)

        return_dict = {
            "xyz": torch.from_numpy(xyz_world).float(),
            "rgb": torch.from_numpy(rgb).float(),
            "xyz_voxel": torch.from_numpy(xyz_voxel).int(),
            "xyz_offset": torch.from_numpy(xyz_offset).float(),
            "bbox": torch.from_numpy(bbox).float(),
            "bbox_voxel": torch.from_numpy(bbox_voxel).int(),
            "transform": torch.from_numpy(transform).float(),
            "scene_id": scene_id,
        }

        if self.return_gt:
            return_dict["xyz_gt"] = torch.from_numpy(xyz_world_gt).float()
            return_dict["xyz_voxel_gt"] = torch.from_numpy(xyz_voxel_gt).int()

            if self.read_normal_gt:
                return_dict["normal_gt"] = torch.from_numpy(normal_gt).float()

        if self.read_normal:
            return_dict["normal"] = torch.from_numpy(normal).float()
        return return_dict

    def load_colmap_points(self, scene_id):
        ply_path = self.ply_path / f"{scene_id}.ply"
        xyz, color = self.load_ply(ply_path)
        return xyz, color

    def load_mesh_points(self, scene_id: str, load_normal: bool):
        ply_path = self.gt_ply_path / f"{scene_id}.ply"
        if load_normal:
            xyz, rgb, normal = self.load_ply(ply_path, read_normal=True)
        else:
            xyz, rgb = self.load_ply(ply_path, read_normal=False)
            normal = None
        return xyz, rgb, normal

    def load_voxelized_colmap_points(self, scene_id, voxel_size):
        xyz, rgb = self.load_colmap_points(scene_id)
        xyz_world = xyz.copy()      # save the original xyz coordinate

        bbox_min = np.min(xyz, axis=0)
        bbox_max = np.max(xyz, axis=0)

        xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, xyz_world = voxelize(
            xyz,
            rgb,
            voxel_size=voxel_size,
            xyz_world=xyz_world,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        bbox_voxel_min = np.min(xyz_voxel, axis=0)
        bbox_voxel_max = np.max(xyz_voxel, axis=0)
        bbox_voxel = np.stack([bbox_voxel_min, bbox_voxel_max], axis=0).astype(np.int32)
        return xyz_world, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel

    def load_voxelized_mesh_points(self, scene_id: str, voxel_size: float, load_normal: bool, bbox_min=None, bbox_max=None):
        if load_normal:
            xyz, rgb, normal = self.load_mesh_points(scene_id, load_normal=load_normal)
            xyz_world = xyz.copy()      # save the original xyz coordinate
            xyz_world = np.concatenate([xyz_world, normal], axis=1)

            _xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, xyz_world = voxelize(
                xyz,
                rgb,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                voxel_size=voxel_size,
                xyz_world=xyz_world,
            )
            normal = xyz_world[:, 3:]
            xyz_world = xyz_world[:, :3]

            normal = normal / np.linalg.norm(normal, axis=1)[:, None].clip(min=1e-9)
        else:
            xyz, rgb, _ = self.load_mesh_points(scene_id, load_normal=load_normal)
            xyz_world = xyz.copy()      # save the original xyz coordinate
            _xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, xyz_world = voxelize(
                xyz,
                rgb,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                voxel_size=voxel_size,
                xyz_world=xyz_world,
            )
            normal = None
        return xyz_world, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, normal

    def load_voxelized_point_pair(self, scene_id: str, voxel_size: float, load_normal: bool):
        # Return the colmap points and mesh points with the same voxelization

        # Load the colmap points
        xyz_input, rgb_input, xyz_voxel_input, _, bbox, bbox_voxel, world_to_voxel_input = self.load_voxelized_colmap_points(scene_id, voxel_size)

        if load_normal:
            xyz, rgb, normal = self.load_mesh_points(scene_id, load_normal=load_normal)
            xyz_world = xyz.copy()      # save the original xyz coordinate
            xyz_world = np.concatenate([xyz_world, normal], axis=1)

            _xyz, rgb_gt, xyz_voxel_gt, _, _, world_to_voxel_gt, xyz_world = voxelize(
                xyz,
                rgb,
                voxel_size=voxel_size,
                xyz_world=xyz_world,
                bbox_min=bbox[0],
                bbox_max=bbox[1],
            )
            normal = xyz_world[:, 3:]
            xyz_world = xyz_world[:, :3]
        else:
            xyz, rgb, _ = self.load_mesh_points(scene_id, load_normal=load_normal)
            xyz_world = xyz.copy()      # save the original xyz coordinate

            _xyz, rgb_gt, xyz_voxel_gt, _, _, world_to_voxel_gt, xyz_world = voxelize(
                xyz,
                rgb,
                voxel_size=voxel_size,
                xyz_world=xyz_world,
                bbox_min=bbox[0],
                bbox_max=bbox[1],
            )
            normal = None
        xyz_gt = xyz_world
        return xyz_input, rgb_input, xyz_voxel_input, xyz_gt, rgb_gt, xyz_voxel_gt, world_to_voxel_input, bbox

    @staticmethod
    def collate_fn(batch):
        # Put all the data into a list
        outputs = {}
        for key in batch[0]:
            outputs[key] = [data[key] for data in batch]
        return outputs
