from typing import List, Optional, Dict, Any
import math
import numpy as np
import torch
import plyfile


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W).astype(np.float32)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY) -> torch.Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def batched_angular_dist_rot_matrix_mvs(
    R1: torch.Tensor,
    R2: torch.Tensor,
    EPS: float = 1e-8
) -> torch.Tensor:
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    diag = torch.diagonal(torch.matmul(R2.permute(0, 2, 1), R1), dim1=1, dim2=2)
    trace = (1 - diag).sum(-1)
    return trace


def get_nearest_camera_ids_mvs(
    target_pose: torch.Tensor,
    ref_poses: torch.Tensor,
    num_views: int,
    ignore_id: Optional[int] = None,
    ignore_mask: Optional[torch.Tensor] = None,
):
    '''Return the indices of the camera poses that are the N-nearest to the target camera pose
        The original implementation use position distance for LLFF.
        However, for indoor scenes, need to consider both position and rotation.

    Args:
        target_pose (torch.Tensor): (3, 4)
        ref_poses (torch.Tensor): (num_views, 3, 4)
        num_views (int): number of nearest views to return
        ignore_id (int, optional): the index of the camera pose to ignore. Defaults to None.
        ignore_mask (torch.Tensor, optional): (num_views,) the mask of the camera poses to ignore . Defaults to None.
    '''
    # target_pose = target_pose[None, ...].repeat(ref_poses.shape[0], 0)
    target_pose = torch.repeat_interleave(target_pose[None, ...], ref_poses.shape[0], dim=0)

    angular_dists = batched_angular_dist_rot_matrix_mvs(target_pose[:, :3, :3], ref_poses[:, :3, :3])
    pos_dists = torch.linalg.norm(target_pose[:, :3, 3] - ref_poses[:, :3, 3], axis=1) ** 2
    if ignore_id is not None:
        assert ignore_id < ref_poses.shape[0], f"ignore_id {ignore_id} is out of range {ref_poses.shape[0]}"
        angular_dists[ignore_id] = torch.inf
        pos_dists[ignore_id] = torch.inf
    if ignore_mask is not None:
        assert ignore_mask.shape[0] == ref_poses.shape[0], f"ignore_mask {ignore_mask.shape[0]} is not equal to ref_poses.shape[0] {ref_poses.shape[0]}"
        angular_dists[ignore_mask] = torch.inf
        pos_dists[ignore_mask] = torch.inf
    dists = angular_dists * 2.0 / 3 + pos_dists
    sorted_ids = torch.argsort(dists)
    return sorted_ids[:num_views]


def save_ply(path, xyz, rgb):
    assert xyz.shape[0] == rgb.shape[0]
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        # print("rgb", rgb.min(), rgb.max())
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    n = xyz.shape[0]
    if rgb.shape[1] == 3:
        rgb = np.concatenate([rgb, np.ones((n, 1), dtype=rgb.dtype) * 255], axis=1)
    vertex = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['red'] = rgb[:, 0]
    vertex['green'] = rgb[:, 1]
    vertex['blue'] = rgb[:, 2]
    vertex['alpha'] = rgb[:, 3]
    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')], text=False)
    ply.write(path)


def save_normal_ply(path, xyz, normal):
    # Convert normal to color
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    normal = (normal + 1) / 2
    normal = (normal * 255).astype(np.uint8)

    n = xyz.shape[0]
    vertex = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['red'] = normal[:, 0]
    vertex['green'] = normal[:, 1]
    vertex['blue'] = normal[:, 2]
    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')], text=False)
    ply.write(path)


def scale_intrinsics(intrinsics: torch.Tensor, scale_ratio_h: float, scale_ratio_w: float):
    """
    Args:
        intrinsics (torch.Tensor): (3, 3)
        scale_ratio_h (float): the ratio to scale the intrinsics
        scale_ratio_w (float): the ratio to scale the intrinsics
    Returns:
        intrinsics (torch.Tensor): (3, 3)
    """
    intrinsics = intrinsics.clone()

    intrinsics[0:2, 2] -= 0.5
    intrinsics[0, 0] *= scale_ratio_w
    intrinsics[1, 1] *= scale_ratio_h

    intrinsics[0, 2] = (intrinsics[0, 2] + 0.5) * scale_ratio_w - 0.5
    intrinsics[1, 2] = (intrinsics[1, 2] + 0.5) * scale_ratio_h - 0.5

    return intrinsics


def scale_intrinsics_np(intrinsics: np.ndarray, scale_ratio_h: float, scale_ratio_w: float):
    """
    Args:
        intrinsics (np.ndarray): (3, 3)
        scale_ratio_h (float): the ratio to scale the intrinsics
        scale_ratio_w (float): the ratio to scale the intrinsics
    Returns:
        intrinsics (np.ndarray): (3, 3)
    """
    intrinsics = intrinsics.copy()

    intrinsics[0:2, 2] -= 0.5
    intrinsics[0, 0] *= scale_ratio_w
    intrinsics[1, 1] *= scale_ratio_h

    intrinsics[0, 2] = (intrinsics[0, 2] + 0.5) * scale_ratio_w - 0.5
    intrinsics[1, 2] = (intrinsics[1, 2] + 0.5) * scale_ratio_h - 0.5

    return intrinsics


def stack_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    outputs = {}
    for k in data[0].keys():
        if isinstance(data[0][k], torch.Tensor):
            outputs[k] = torch.stack([x[k] for x in data], dim=0)
        elif isinstance(data[0][k], np.ndarray):
            outputs[k] = np.stack([x[k] for x in data], axis=0)
        else:
            outputs[k] = [x[k] for x in data]
    return outputs


def random_sample(
    sample_mask: np.ndarray,
    num_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random sample non-repeated indices from the valid samples

    Args:
        sample_mask (np.ndarray): boolean mask of the samples. 1 for valid samples, 0 for invalid samples
        num_samples (int): number of samples to select
    Returns:
        indices (np.ndarray): (num_samples,) tensor of indices
    """
    indices = np.where(sample_mask)[0]
    if len(indices) < num_samples:
        raise ValueError(f"Number of valid samples: {len(indices)} is less than num_samples to sample: {num_samples}")
    elif len(indices) == num_samples:
        return indices

    if generator is None:
        return np.random.choice(indices, num_samples, replace=False)
    else:
        return generator.choice(indices, num_samples, replace=False)


def farthest_point_sampling(
    xyz: np.ndarray,
    num_points: int,
    invalid_mask: Optional[np.ndarray] = None,
    generator: Optional[np.random.Generator] = None,
    randomness: bool = False,
    random_topk: int = 10,
) -> np.ndarray:
    """Farthest point sampling algorithm to sample num_points from xyz

    Args:
        xyz: (N, 3) tensor of points
        num_points: number of points to sample
        invalid_mask: (N,) tensor of invalid points

    Returns:
        indices: (num_points,) tensor of indices
    """
    num_points = min(num_points, xyz.shape[0])
    num_valid_points = np.sum(~invalid_mask)
    assert num_valid_points >= num_points, f"Number of valid points: {num_valid_points} is less than num_points to sample: {num_points}"
    indices = np.zeros(num_points, dtype=np.int64)
    # Randomly choose the first point that is not invalid
    if generator is None:
        indices[0] = np.random.choice(np.where(~invalid_mask)[0])
    else:
        indices[0] = generator.choice(np.where(~invalid_mask)[0])
    # Compute the distance to the first point
    dist = np.sum((xyz - xyz[indices[0]])**2, axis=1)
    dist[invalid_mask] = -1e8
    for i in range(1, num_points):
        if not randomness:
            # Find the farthest point
            farthest = np.argmax(dist)
        else:
            # Randomly sample topk points and choose the farthest one
            topk = np.argsort(dist)[-random_topk:]
            # Every instance in topk should be valid
            assert not np.any(invalid_mask[topk]), f"farthest point in topk is invalid {invalid_mask[topk]}"
            if generator is None:
                # farthest = topk[np.random.choice(random_topk)]
                farthest = np.random.choice(topk)
            else:
                farthest = generator.choice(topk)

        # Update the distance
        dist = np.minimum(dist, np.sum((xyz - xyz[farthest]) ** 2, axis=1))
        dist[invalid_mask] = -1e8
        # Update the indices
        indices[i] = farthest
        assert not invalid_mask[farthest], f"farthest point {farthest} is invalid"
    return indices


def save_mesh_ply(path, vertices, faces=None):
    n = vertices.shape[0]
    vertex = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = vertices[:, 0]
    vertex['y'] = vertices[:, 1]
    vertex['z'] = vertices[:, 2]
    if faces is None:
        ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')], text=False)
    else:
        m = faces.shape[0]
        face = np.zeros(m, dtype=[('vertex_indices', 'i4', (3,))])
        face['vertex_indices'] = faces
        ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex'), plyfile.PlyElement.describe(face, 'face')], text=False)
    ply.write(path)


class RepeatSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        batch_size: int,
        num_repeat: int,
        shuffle: bool = True,
        generator: Optional[np.random.Generator] = None,
        # seed: Optional[int] = None,
    ):
        """
        If the input is [1, 2, 3, 4, 5], batch_size is 2 and num_repeat is 3, the output will be:
        [1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 1], [5, 1], [5, 1]
        """
        self.data_source = data_source
        self.num_repeat = num_repeat
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        if generator is not None:
            self.generator = generator

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            if self.generator is not None:
                self.generator.shuffle(indices)
            else:
                np.random.shuffle(indices)

        if len(indices) % self.batch_size != 0:
            # Pad the indices to make it a multiple of batch_size
            indices = np.concatenate([indices, indices[:self.batch_size - len(indices) % self.batch_size]])

        # Group by batch_size
        batched_indices = indices.reshape(-1, self.batch_size)
        # Repeat the indices
        batched_indices = np.repeat(batched_indices, self.num_repeat, axis=0)
        indices = batched_indices.flatten()
        return iter(indices)

    def __len__(self):
        return len(self.data_source) * self.num_repeat


def dilation_sparse(x: torch.Tensor, radius: int, unique: bool = True) -> torch.Tensor:
    # x: (N, 3) or (N, 4)
    x = x.int()
    dim = x.shape[1]

    if dim == 3:
        grid = torch.tensor([
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ], device=x.device, dtype=torch.int32).reshape(1, -1, 3)  # 1, 7, 3
    elif dim == 4:
        grid = torch.tensor([
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [0, 0, 0, 1],
        ], device=x.device, dtype=torch.int32).reshape(1, -1, 4)  # 1, 7, 3

    for _ in range(radius):
        x = x.unsqueeze(1)  # (N, 1, 3) or (N, 1, 4)
        x = x + grid  # (N, 7, 3) or (N, 7, 4)

        # Get the unique points
        x = x.reshape(-1, dim)  # (N * 7, 3) or (N * 7, 4)
        if unique:
            x = torch.unique(x, dim=0)  # (M, 3) or (M, 4)
    return x


def apply_transform(xyz, matrix):
    # xyz: (N, 3)
    # matrix: (4, 4)

    xyz_out = matrix[:3, :3] @ xyz.T + matrix[:3, 3:4]
    return xyz_out.T


def apply_rotation(xyz, matrix):
    matrix = matrix[:3, :3]
    # Remove the scale: this assume the scaling is isotropic
    matrix = matrix / np.linalg.norm(matrix[:, 0])
    xyz_out = matrix @ xyz.T
    return xyz_out.T


def get_translation_matrix(translation):
    # translation: (3,)
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return matrix


def voxelize(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
    xyz_world: Optional[np.ndarray] = None,
):
    """Voxelize the point cloud. Crop the point cloud to the bounding box. Remove duplicates in the same voxel.

    Returns:
        xyz: (N, 3) point coordinates
        rgb: (N, 3) point colors
        xyz_voxel: (N, 3) voxel coordinates (int)
        xyz_offset: (N, 3) offset from the voxel coordinates. assuming the voxel is at the center of the voxel
        bbox: (2, 3) bounding
        transform: (4, 4) transformation matrix that does voxelization (from world to voxel space)
    """

    if bbox_min is None:
        bbox_min = np.min(xyz, axis=0)
    if bbox_max is None:
        bbox_max = np.max(xyz, axis=0)

    # Remove points outside the bounding box
    mask = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    # print("Remove points outside the bounding box:", np.sum(~mask))
    xyz = xyz[mask]

    # The voxelization transform (xyz - bbox_min) / voxel_size
    transform = np.eye(4)
    transform[:3, 3] = -bbox_min / voxel_size
    transform[:3, :3] *= 1 / voxel_size

    xyz_voxel = apply_transform(xyz, transform.astype(np.float32))
    xyz_voxel_int = np.floor(xyz_voxel).astype(np.int32)
    # The offset is between [-0.5, 0.5]
    xyz_offset = (xyz_voxel - xyz_voxel_int - 0.5)

    # The voxel center should be +0.5
    shift_transform = np.eye(4)
    shift_transform[:3, 3] = -0.5
    transform = shift_transform @ transform

    # Remove duplicates in the same voxel
    xyz_voxel_int, mapping = np.unique(xyz_voxel_int, axis=0, return_index=True)
    xyz = xyz[mapping]
    xyz_offset = xyz_offset[mapping]

    # Prepare output
    bbox = np.array([bbox_min, bbox_max])
    rgb = rgb[mask][mapping]
    if xyz_world is not None:
        # Do the same thing as rgb
        xyz_world = xyz_world[mask][mapping]
        return xyz, rgb, xyz_voxel_int, xyz_offset, bbox, transform, xyz_world

    return xyz, rgb, xyz_voxel_int, xyz_offset, bbox, transform
