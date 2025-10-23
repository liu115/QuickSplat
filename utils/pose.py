import torch
import pytorch3d.transforms as transforms3d


def apply_transform(xyz, matrix):
    # xyz: (..., 3)
    # matrix: (4, 4)
    xyz_out = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    return xyz_out


def apply_rotation(xyz, matrix):
    # xyz: (..., 3)
    # matrix: (3, 3)
    xyz_out = xyz @ matrix.T
    return xyz_out


def quaternion_to_normal(quat):
    # TODO: Check if this is correct in 2DGS
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    matrix = transforms3d.quaternion_to_matrix(quat)
    normal = matrix[:, :, 2]    # <- this is correct
    # normal = matrix[:, 2, :]
    return normal
