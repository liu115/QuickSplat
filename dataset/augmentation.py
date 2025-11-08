"""point cloud augmentation functions using pytorch"""
import torch
import torch.nn.functional as F
import numpy as np


def random_perturb(xyz, rgb, noise_level=0.01, clip=0.05):
    """Randomly perturb the point cloud by adding noise to each point"""
    noise = torch.randn_like(xyz) * noise_level
    noise = torch.clamp(noise, -clip, clip)
    xyz = xyz + noise
    return xyz, rgb


def random_perturb_np(xyz, rgb, noise_level=0.01, clip=0.05):
    """Randomly perturb the point cloud by adding noise to each point"""
    noise = np.random.randn(*xyz.shape).astype(xyz.dtype) * noise_level
    noise = np.clip(noise, -clip, clip)
    xyz = xyz + noise
    return xyz, rgb


def random_dropout(xyz, rgb, dropout_ratio_range=(0.1, 0.25)):
    """Randomly drop fixed amount of points from the point cloud"""
    ratio = np.random.uniform(*dropout_ratio_range)
    perm = torch.randperm(xyz.shape[0])
    # print(f"Before dropout: {xyz.shape[0]}")
    num_points = int(xyz.shape[0] * (1 - ratio))
    xyz = xyz[perm[:num_points]]
    rgb = rgb[perm[:num_points]]
    # print(f"After dropout: {xyz.shape[0]}")
    return xyz, rgb


def random_dropout_np(xyz, rgb, dropout_ratio_range=(0.1, 0.25)):
    """Randomly drop fixed amount of points from the point cloud"""
    ratio = np.random.uniform(*dropout_ratio_range)
    perm = np.random.permutation(xyz.shape[0])
    # print(f"Before dropout: {xyz.shape[0]}")
    num_points = int(xyz.shape[0] * (1 - ratio))
    xyz = xyz[perm[:num_points]]
    rgb = rgb[perm[:num_points]]
    # print(f"After dropout: {xyz.shape[0]}")
    return xyz, rgb


def random_add(xyz, rgb, add_ratio_range=(0.1, 0.25), aabb=None):
    """Randomly add fixed amount of points into the point cloud"""
    ratio = np.random.uniform(*add_ratio_range)
    num_points = int(xyz.shape[0] * ratio)
    if aabb is not None:
        xyz_min = aabb[0]
        xyz_max = aabb[1]
    else:
        xyz_min = xyz.min(dim=0)[0]
        xyz_max = xyz.max(dim=0)[0]

    new_xyz = torch.rand((num_points, 3), device=xyz.device)
    new_xyz = new_xyz * (xyz_max - xyz_min) + xyz_min
    new_rgb = torch.rand((num_points, 3), device=rgb.device)
    xyz = torch.cat([xyz, new_xyz], dim=0)
    rgb = torch.cat([rgb, new_rgb], dim=0)
    return xyz, rgb


def random_add_np(xyz, rgb, add_ratio_range=(0.1, 0.25), aabb=None):
    """Randomly add fixed amount of points into the point cloud"""
    ratio = np.random.uniform(*add_ratio_range)
    num_points = int(xyz.shape[0] * ratio)
    if aabb is not None:
        xyz_min = aabb[0]
        xyz_max = aabb[1]
    else:
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
    # Make aabb a bit larger (1.05x)
    size = (xyz_max - xyz_min)
    xyz_min = xyz_min - size * 0.025
    xyz_max = xyz_max + size * 0.025

    new_xyz = np.random.rand(num_points, 3).astype(xyz.dtype)
    new_xyz = new_xyz * (xyz_max - xyz_min) + xyz_min
    new_rgb = np.random.rand(num_points, 3).astype(rgb.dtype)
    xyz = np.concatenate([xyz, new_xyz], axis=0)
    rgb = np.concatenate([rgb, new_rgb], axis=0)
    return xyz, rgb


def random_rotate_points(xyz, xyz_gt=None):
    # Rotate the point cloud along z-axis
    angle = np.random.uniform(0, 2 * np.pi)
    matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    xyz = np.dot(xyz, matrix)
    if xyz_gt is not None:
        xyz_gt = np.dot(xyz_gt, matrix)
        return xyz, xyz_gt, angle

    return xyz, angle


def random_flip_points(xyz, xyz_gt=None):
    # Flip the point cloud x-axis or y-axis

    flip_x = np.random.choice([True, False])
    flip_y = np.random.choice([True, False])
    new_xyz = xyz.copy()

    if flip_x:
        new_xyz[:, 0] = -new_xyz[:, 0]

    if flip_y:
        new_xyz[:, 1] = -new_xyz[:, 1]

    if xyz_gt is not None:
        new_xyz_gt = xyz_gt.copy()

        if flip_x:
            new_xyz_gt[:, 0] = -new_xyz_gt[:, 0]

        if flip_y:
            new_xyz_gt[:, 1] = -new_xyz_gt[:, 1]

        return new_xyz, new_xyz_gt, flip_x, flip_y


def random_global_3d_aug(rot90_only=False):
    # Generate a random augmentation (4 x 4 matrix) that includes flipping, rotation, scaling, and translation
    # The rotation is around the z-axis
    # Flip the point cloud x-axis or y-axis
    # Scaling is isotropic within range [0.8, 1.2]
    # Translation is within range [-0.1, 0.1] for each axis

    flip_x = np.random.choice([True, False])
    flip_y = np.random.choice([True, False])
    if rot90_only:
        angle = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    else:
        angle = np.random.uniform(0, 2 * np.pi)
    scale = np.random.uniform(0.8, 1.2)
    translation = np.random.uniform(-0.1, 0.1, size=3)

    flip_matrix = np.eye(4, dtype=np.float32)
    if flip_x:
        flip_matrix[0, 0] = -1
    if flip_y:
        flip_matrix[1, 1] = -1

    rotate_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[0, 0] = scale
    scale_matrix[1, 1] = scale
    scale_matrix[2, 2] = scale

    translation_matrix = np.eye(4, dtype=np.float32)
    translation_matrix[:3, 3] = translation

    aug_matrix = translation_matrix @ scale_matrix @ rotate_matrix @ flip_matrix
    return aug_matrix


def random_crop_xy(xyz, xyz_gt, crop_size: float, min_points=0, max_trial=10):
    xyz_min = np.min(xyz_gt, axis=0)
    xyz_max = np.max(xyz_gt, axis=0)

    center = (xyz_min + xyz_max) / 2
    xyz_min = center - (center - xyz_min) * 1.2
    xyz_max = center + (xyz_max - center) * 1.2

    # x_range = xyz_max[0] - xyz_min[0]
    # y_range = xyz_max[1] - xyz_min[1]

    # Randomly crop the point cloud on the xy-plane
    start_x = np.random.uniform(xyz_min[0], xyz_max[0] - crop_size)
    start_y = np.random.uniform(xyz_min[1], xyz_max[1] - crop_size)

    mask_gt = (xyz_gt[:, 0] >= start_x) & (xyz_gt[:, 0] <= start_x + crop_size) & \
              (xyz_gt[:, 1] >= start_y) & (xyz_gt[:, 1] <= start_y + crop_size)
    mask = (xyz[:, 0] >= start_x) & (xyz[:, 0] <= start_x + crop_size) & \
           (xyz[:, 1] >= start_y) & (xyz[:, 1] <= start_y + crop_size)

    if min_points > 0 and max_trial > 0:
        if mask.sum() < min_points or mask_gt.sum() < min_points:
            return random_crop_xy(xyz, xyz_gt, crop_size, min_points, max_trial - 1)

    return mask, mask_gt
