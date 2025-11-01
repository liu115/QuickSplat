from typing import Optional, Literal, Tuple, Dict, Union
from simple_knn._C import distCUDA2

import numpy as np
import torch
import torch.nn as nn
import pytorch3d.transforms as transforms3d

from modules.mlp import MLP
from utils.pose import apply_transform, apply_rotation
# from models.gaussian_utils import inverse_sigmoid


MAX_LOG_SCALE = 9         # 1e2 after exp * voxel_size (0.02)
MIN_LOG_SCALE = -9       # 2e-6 after exp * voxel_size (0.02)
MIN_BEFORE_SIGMOID = -15
MAX_BEFORE_SIGMOID = 10


def inverse_sigmoid(x, eps=1e-6):
    denom = (1 - x).clamp(min=eps)
    y = torch.log((x / denom).clamp(min=eps))
    return y


def inverse_softplus(x, beta=1.0, threshold=20.0):
    # Compute the inverse of softplus
    # return torch.log(torch.exp(x * beta) - 1) / beta
    output = torch.zeros_like(x)
    mask = x > threshold
    if mask.any():
        output[mask] = x[mask]
        output[~mask] = torch.log(torch.exp(x[~mask] * beta) - 1) / beta
    else:
        output = torch.log(torch.exp(x * beta) - 1) / beta
    return output


# def apply_transform(xyz, matrix):
#     # xyz: (..., 3)
#     # matrix: (4, 4)

#     # xyz_out = matrix[:3, :3] @ xyz.T + matrix[:3, 3:4]
#     # return xyz_out.T
#     shape = xyz.shape
#     xyz = xyz.view(-1, 3)
#     xyz_out = xyz @ matrix[:3, :3].T + matrix[:3, 3]
#     xyz_out = xyz_out.view(*shape)
#     return xyz_out


def compute_rotation_from_normal(normal):
    # Compute the rotation from the normal
    # normal: (N, 3)
    # Returns: (N, 9)
    A = torch.zeros_like(normal)
    A[:, 0] = 1.0

    V1 = torch.cross(A, normal, dim=-1)
    # make sure V1 is not zero
    mask = torch.norm(V1, dim=-1) < 1e-6
    if mask.any():
        B = torch.zeros_like(normal)
        B[:, 1] = 1.0
        V1[mask] = torch.cross(B[mask], normal[mask], dim=-1)

    mask = torch.norm(V1, dim=-1) < 1e-6
    if mask.any():
        C = torch.zeros_like(normal)
        C[:, 2] = 1.0
        V1[mask] = torch.cross(C[mask], normal[mask], dim=-1)

    V1 = torch.nn.functional.normalize(V1, dim=-1)
    V2 = torch.cross(normal, V1, dim=-1)
    V2 = torch.nn.functional.normalize(V2, dim=-1)

    rotation = torch.cat((V1, V2, normal), dim=-1)
    return rotation


class ScaffoldGSFull(nn.Module):
    def __init__(self, config, voxel_size, bbox):
        super().__init__()
        self.bbox = bbox
        self.voxel_size = voxel_size

        self.config = config.GSPLAT
        self.scaffold_config = config.SCAFFOLD
        # self.active_sh_degree = self.config.init_sh_degree
        # self.max_sh_degree = self.config.max_sh_degree

        self._latent = None
        self.unit_scale_multiplier = 1.0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def get_raw_params(self):
        if self._latent is None:
            raise ValueError("Please use ScaffoldGSFull.create_from_latent() or ScaffoldGSFull.create_from_voxel2() to initialize the model")
        return {"latent": self._latent}

    def get_raw_opacity(self):
        return torch.sigmoid(self._latent[:, 10:11])

    def set_raw_params(self, params):
        self._latent = nn.Parameter(params["latent"], requires_grad=True)

    def set_raw_params_and_xyz(self, params, xyz_voxel):
        self._latent = nn.Parameter(params["latent"], requires_grad=True)
        self.xyz_voxel = nn.Parameter(xyz_voxel, requires_grad=False)

    @property
    def num_points(self):
        return self._latent.shape[0]

    def reset_scaffold(
        self,
        latent: torch.Tensor,
        xyz_voxel: torch.Tensor,
    ):
        assert latent.shape[0] == xyz_voxel.shape[0]
        self._latent = nn.Parameter(latent, requires_grad=True)

        self.xyz_voxel = nn.Parameter(xyz_voxel.int(), requires_grad=False)
        # Recompute the center of the voxel based on xyz_voxel, self.bbox, self.voxel_size
        xyz_voxel_center = (xyz_voxel.float() + 0.5) * self.voxel_size + self.bbox[0]
        self.xyz_voxel_center = nn.Parameter(xyz_voxel_center, requires_grad=False)
        self.max_radii2D = torch.zeros((self._latent.shape[0]), device="cuda")

    def concat_new_voxels(
        self,
        xyz_voxel: torch.Tensor,
        rgb: torch.Tensor,
        scale: torch.Tensor,
        opacity: torch.Tensor,
        rotation: torch.Tensor,
    ):
        # xyz_voxel: (N, 3)
        # rgb: (N, 3)
        # scale: (N, 3)
        # opacity: (N, 1)
        # rotation: (N, 4)
        xyz_offset = torch.zeros((xyz_voxel.shape[0], 3), device="cuda")
        xyz_offset = inverse_sigmoid((xyz_offset + 1) / 2)

        # if self.scaffold_config.exp_scale:
        if self.scaffold_config.scale_activation == "exp":
            d = scale / self.voxel_size
            scales = torch.log(d)[..., None].repeat(1, 3)
            scales = torch.clamp(scales, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
        elif self.scaffold_config.scale_activation == "softplus":
            d = scale / self.voxel_size
            d = inverse_softplus(d)
            d = torch.clamp(d, max=1e2 * self.voxel_size)
            scales = d[..., None].repeat(1, 3)
        elif self.scaffold_config.scale_activation == "abs":
            d = scale / self.voxel_size
            scales = d[..., None].repeat(1, 3)
        else:   # sigmoid
            max_scale = self.scaffold_config.max_scale
            d = scale / (self.voxel_size * max_scale)
            d = torch.clamp(d, min=0.0, max=1.0)
            scales = inverse_sigmoid(d)[..., None].repeat(1, 3)

        opacity = inverse_sigmoid(opacity)
        rgb = inverse_sigmoid(rgb)
        # print(xyz_offset.shape, scales.shape, rotation.shape, opacity.shape, rgb.shape)
        # breakpoint()

        new_latent = torch.zeros((xyz_voxel.shape[0], self._latent.shape[1]), device="cuda")
        new_latent[:, :3] = xyz_offset
        new_latent[:, 3:6] = scales
        if self.is_2dgs:
            new_latent[:, 5] = 0.0  # The third scale dimension doesn't matter
        new_latent[:, 6:10] = rotation
        new_latent[:, 10:11] = opacity
        new_latent[:, 11:14] = rgb

        self._latent = nn.Parameter(torch.cat([self._latent, new_latent], dim=0), requires_grad=True)
        self.xyz_voxel = nn.Parameter(torch.cat([self.xyz_voxel, xyz_voxel.int()], dim=0), requires_grad=False)
        return new_latent

    @staticmethod
    def create_from_latent(
        config,
        xyz_voxel: torch.Tensor,
        latent: torch.Tensor,
        bbox: torch.Tensor,
        voxel_to_world_transform: torch.Tensor,
        voxel_size: float,
        spatial_lr_scale: float,
    ):
        model = ScaffoldGSFull(config, voxel_size, bbox)
        model._latent = nn.Parameter(latent, requires_grad=True)
        model.xyz_voxel = nn.Parameter(xyz_voxel, requires_grad=False)
        model.transform = nn.Parameter(voxel_to_world_transform, requires_grad=False)
        model.spatial_lr_scale = spatial_lr_scale
        return model

    @staticmethod
    def create_from_voxels2(
        config,
        hidden_dim: int,
        voxel_size: float,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        xyz_voxel: torch.Tensor,
        xyz_offset: torch.Tensor,
        transform: torch.Tensor,
        bbox: torch.Tensor,
        spatial_lr_scale: float,
        zero_latent: bool = False,
        normal: Optional[torch.Tensor] = None,
        is_2dgs: bool = False,
        unit_scale: bool = False,
        unit_scale_multiplier: float = 1.0,
        seed: Optional[int] = None,
    ):
        if unit_scale:
            dist = torch.ones_like(xyz[:, 0]) * voxel_size * unit_scale_multiplier
        else:
            dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
            dist = torch.sqrt(dist2)

        # scales = torch.log(torch.sqrt(dist2) / voxel_size)[..., None].repeat(1, 3)
        # if config.SCAFFOLD.exp_scale:
        if config.SCAFFOLD.scale_activation == "exp":
            d = dist / voxel_size
            scales = torch.log(d)[..., None].repeat(1, 3)
            scales = torch.clamp(scales, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
        elif config.SCAFFOLD.scale_activation == "softplus":
            dist = torch.clamp(dist, min=1e-8, max=1e2)
            d = dist / voxel_size
            d = inverse_softplus(d)
            scales = d[..., None].repeat(1, 3)
        elif config.SCAFFOLD.scale_activation == "abs":
            d = dist / voxel_size
            scales = d[..., None].repeat(1, 3)
        else:   # sigmoid
            max_scale = config.SCAFFOLD.max_scale
            d = dist / (voxel_size * max_scale)
            d = torch.clamp(d, min=0.0, max=1.0)
            scales = inverse_sigmoid(d)[..., None].repeat(1, 3)

        if normal is None:
            rots = torch.zeros((xyz.shape[0], 4), device="cuda")
            rots[:, 0] = 1
        else:
            # print("using normal")
            # print(normal.shape)
            rot_mat = compute_rotation_from_normal(normal)
            rot_mat = rot_mat.view(-1, 3, 3)
            rots = transforms3d.matrix_to_quaternion(rot_mat)

        opacities = inverse_sigmoid(config.GSPLAT.init_opacity * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        rgb = inverse_sigmoid(rgb)

        model = ScaffoldGSFull(config, voxel_size, bbox)
        model.unit_scale = unit_scale
        model.unit_scale_multiplier = unit_scale_multiplier
        model.is_2dgs = is_2dgs
        model.spatial_lr_scale = spatial_lr_scale
        model.xyz_voxel = nn.Parameter(xyz_voxel, requires_grad=False)
        model.transform = nn.Parameter(transform, requires_grad=False)

        xyz_offset = xyz_offset / config.SCAFFOLD.offset_scaling
        # From [-1.0, 1.0] to [0.0, 1.0] to real space
        xyz_offset = inverse_sigmoid((xyz_offset + 1) / 2)

        assert zero_latent is True
        latent = torch.zeros((xyz.shape[0], hidden_dim), device="cuda")

        if is_2dgs:
            raise NotImplementedError("Fix this TODO")
            latent[:, :3] = xyz_offset
            latent[:, 3:6] = scales
            latent[:, 5] = 0.0  # The third scale dimension doesn't matter
            latent[:, 6:10] = rots
            latent[:, 10:11] = opacities
            latent[:, 11:14] = rgb
        else:
            latent[:, :3] = xyz_offset
            latent[:, 3:6] = scales
            latent[:, 6:10] = rots
            latent[:, 10:11] = opacities
            latent[:, 11:14] = rgb

        model._latent = nn.Parameter(latent, requires_grad=True)
        return model

    @torch.no_grad()
    def get_point_cloud(self, transform):
        xyz_voxel = self.xyz_voxel
        xyz_offset = self._latent[:, :3]
        xyz_offset = torch.sigmoid(xyz_offset) * 2.0 - 1

        xyz = xyz_voxel.float() + xyz_offset
        # Apply the 4x4 transform
        xyz = transform[:3, :3] @ xyz.t() + transform[:3, 3:4]
        xyz = xyz.t()

        rgb = self._latent[:, 11:14]
        rgb = torch.sigmoid(rgb)
        return xyz, rgb

    @staticmethod
    def create_from_voxels(
        config,
        hidden_dim: int,
        voxel_size: float,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        xyz_voxel: torch.Tensor,
        bbox: torch.Tensor,
        spatial_lr_scale: float,
        zero_latent: bool = False,
        seed: Optional[int] = None,
    ):
        raise NotImplementedError("Use create_from_voxels2 instead")

        # assert xyz.device == torch.device("cpu")
        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2) / voxel_size)[..., None].repeat(1, 3)
        if config.SCAFFOLD.exp_scale:
            d = torch.sqrt(dist2) / voxel_size
            scales = torch.log(d)[..., None].repeat(1, 3)
            scales = torch.clamp(scales, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
        else:
            max_scale = config.SCAFFOLD.max_scale
            d = torch.sqrt(dist2) / (voxel_size * max_scale)
            d = torch.clamp(d, min=0.0, max=1.0)
            scales = inverse_sigmoid(d)[..., None].repeat(1, 3)
            # scales = inverse_sigmoid(d)[..., None].repeat(1, 3) + scale_offset

        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(config.GSPLAT.init_opacity * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        model = ScaffoldGSFull(config, voxel_size, bbox)
        model.spatial_lr_scale = spatial_lr_scale
        model.xyz_voxel_center = (xyz_voxel.float() + 0.5) * voxel_size + bbox[0]

        model.xyz_voxel = nn.Parameter(xyz_voxel, requires_grad=False)
        model.xyz_voxel_center = nn.Parameter(model.xyz_voxel_center, requires_grad=False)

        xyz_offset = (xyz - model.xyz_voxel_center) / voxel_size
        xyz_offset = (xyz_offset + 1) / 2
        xyz_offset = inverse_sigmoid(xyz_offset)
        rgb = inverse_sigmoid(rgb)

        if seed is not None:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
        else:
            generator = None
        if config.SCAFFOLD.skip_connect:
            latent = torch.cat([
                xyz_offset,
                scales,
                rots,
                opacities,
                rgb,
            ], dim=1)

            latent_size = latent.shape[1]
            if not zero_latent:
                latent = torch.cat([
                    latent,
                    torch.randn((latent.shape[0], hidden_dim - latent_size), device="cuda", generator=generator)
                ], dim=1)
            else:
                latent = torch.cat([
                    latent,
                    torch.zeros((latent.shape[0], hidden_dim - latent_size), device="cuda")
                ], dim=1)
        else:
            if not zero_latent:
                latent = torch.randn((xyz.shape[0], hidden_dim), device="cuda", generator=generator)
            else:
                latent = torch.zeros((xyz.shape[0], hidden_dim), device="cuda")

        model._latent = nn.Parameter(latent, requires_grad=True)
        model.max_radii2D = torch.zeros((latent.shape[0]), device="cuda")
        return model

    @staticmethod
    def create_from_points(
        config,
        hidden_dim: int,
        voxel_size: float,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        spatial_lr_scale: float,
        knn_dist2: Optional[torch.Tensor] = None,
        zero_latent: bool = False,
        seed: Optional[int] = None,
    ):
        raise NotImplementedError("Use create_from_voxels2 instead")
        # xyz device should be cpu
        assert xyz.device == torch.device("cpu")

        bbox_min = xyz.min(dim=0).values
        bbox_max = xyz.max(dim=0).values
        bbox = torch.stack([bbox_min, bbox_max], dim=0)     # (2, 3)

        # Map xyz into voxel_grid defined by voxel_size and bbox
        xyz_voxel = (xyz - bbox[0]) / voxel_size
        xyz_voxel = xyz_voxel.int()

        # Remove duplicates and record the select indices
        xyz_voxel, mapping = np.unique(xyz_voxel.numpy(), axis=0, return_index=True)
        xyz_voxel = torch.from_numpy(xyz_voxel).cuda()
        mapping = torch.from_numpy(mapping)
        xyz = (xyz[mapping]).cuda()
        rgb = (rgb[mapping]).cuda()
        bbox = bbox.cuda()

        if knn_dist2 is None:
            dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        else:
            dist2 = torch.clamp_min(knn_dist2, 0.0000001).cuda()

        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        if config.SCAFFOLD.exp_scale:
            d = torch.sqrt(dist2) / voxel_size
            scales = torch.log(d)[..., None].repeat(1, 3)
            scales = torch.clamp(scales, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
        else:
            max_scale = config.SCAFFOLD.max_scale
            d = torch.sqrt(dist2) / (voxel_size * max_scale)
            d = torch.clamp(d, min=0.0, max=1.0)
            scales = inverse_sigmoid(d)[..., None].repeat(1, 3)
            # scales = inverse_sigmoid(d)[..., None].repeat(1, 3) + scale_offset

        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(config.GSPLAT.init_opacity * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        model = ScaffoldGSFull(config, voxel_size, bbox)
        model.spatial_lr_scale = spatial_lr_scale
        # Compute xyz offsets to the center of the voxel
        model.xyz_voxel_center = (xyz_voxel.float() + 0.5) * voxel_size + bbox[0]

        # Move xyz_voxel to positive space
        xyz_voxel = xyz_voxel - xyz_voxel.min(dim=0).values
        model.xyz_voxel = nn.Parameter(xyz_voxel, requires_grad=False)
        model.xyz_voxel_center = nn.Parameter(model.xyz_voxel_center, requires_grad=False)
        xyz_offset = (xyz - model.xyz_voxel_center) / voxel_size

        xyz_offset = (xyz_offset + 1) / 2
        xyz_offset = inverse_sigmoid(xyz_offset)
        rgb = inverse_sigmoid(rgb)

        if seed is not None:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
        else:
            generator = None
        if config.SCAFFOLD.skip_connect:
            latent = torch.cat([
                xyz_offset,
                scales,
                rots,
                opacities,
                rgb,
            ], dim=1)

            latent_size = latent.shape[1]
            if not zero_latent:
                latent = torch.cat([
                    latent,
                    torch.randn((latent.shape[0], hidden_dim - latent_size), device="cuda", generator=generator)
                ], dim=1)
            else:
                latent = torch.cat([
                    latent,
                    torch.zeros((latent.shape[0], hidden_dim - latent_size), device="cuda")
                ], dim=1)
        else:
            if not zero_latent:
                latent = torch.randn((xyz.shape[0], hidden_dim), device="cuda", generator=generator)
            else:
                latent = torch.zeros((xyz.shape[0], hidden_dim), device="cuda")

        model._latent = nn.Parameter(latent, requires_grad=True)
        model.max_radii2D = torch.zeros((latent.shape[0]), device="cuda")
        return model


# class GSIdentityDecoder(nn.Module):
#     def __init__(
#         self,
#         exp_scale: bool = True,
#         max_scale: float = 100.0,
#         scale_activation: Literal["sigmoid", "exp", "softplus", "abs"] = "sigmoid",
#         offset_scaling: float = 1.0,
#     ):
#         super().__init__()
#         self.exp_scale = exp_scale
#         self.max_scale = max_scale
#         self.scale_activation = scale_activation
#         self.offset_scaling = offset_scaling

#     def forward(self, latent, xyz_voxel_center, voxel_size):
#         raise NotImplementedError
#         # xyz = latent[:, :3][:, None, :]
#         # scale = latent[:, 3:6][:, None, :]
#         # rotation = latent[:, 6:10][:, None, :]
#         # opacity = latent[:, 10:11][:, None, :]
#         # rgb = latent[:, 11:14][:, None, :]

#         # # Run activation
#         # # xyz = torch.clamp(xyz, max=MAX_BEFORE_SIGMOID, min=MIN_BEFORE_SIGMOID)
#         # xyz_offset = torch.sigmoid(xyz) * 2.0 - 1
#         # if isinstance(voxel_size, torch.Tensor):
#         #     voxel_size = voxel_size[None, None, :]
#         # xyz = xyz_offset * voxel_size + xyz_voxel_center[:, None, :]
#         # scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
#         # scale = torch.exp(scale) * voxel_size
#         # rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)
#         # # opacity = torch.clamp(opacity, max=MAX_BEFORE_SIGMOID, min=MIN_BEFORE_SIGMOID)
#         # opacity = torch.sigmoid(opacity)
#         # # rgb = torch.clamp(rgb, max=MAX_BEFORE_SIGMOID, min=MIN_BEFORE_SIGMOID)
#         # rgb = torch.sigmoid(rgb)

#         # return {
#         #     "xyz": xyz,
#         #     "scale": scale,
#         #     "rotation": rotation,
#         #     "opacity": opacity,
#         #     "rgb": rgb,
#         # }


# class GSDecoder(nn.Module):
#     def forward(
#         self,
#         latent: torch.Tensor,
#         xyz_voxel_center: torch.Tensor,
#         voxel_size: Union[float, torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         """
#         latent: (N, hidden_dim)
#         xyz_voxel_center: (N, 3)
#         voxel_size: float or (3,)
#         """
#         raise NotImplementedError("deprecated")
#         out = self.mlp(latent)
#         xyz = self.xyz_head(out).reshape(-1, self.num_gs, 3)
#         scale = self.scale_head(out).reshape(-1, self.num_gs, 3)
#         rotation = self.rotation_head(out).reshape(-1, self.num_gs, 4)
#         opacity = self.opacity_head(out).reshape(-1, self.num_gs, 1)
#         rgb = self.rgb_head(out).reshape(-1, self.num_gs, 3)

#         if self.skip_connect:
#             xyz = torch.tanh(xyz)
#             scale = torch.tanh(scale)
#             opacity = torch.tanh(opacity)
#             rgb = torch.tanh(rgb)

#             xyz = xyz + latent[:, :3][:, None, :]
#             scale = scale + latent[:, 3:6][:, None, :]
#             opacity = opacity + latent[:, 10:11][:, None, :]
#             rgb = rgb + latent[:, 11:14][:, None, :]

#             if self.quat_rotation:
#                 rotation = nn.functional.normalize(rotation, p=2, dim=-1)
#                 rotation_input = latent[:, 6:10][:, None, :]
#                 rotation_input = nn.functional.normalize(rotation_input, p=2, dim=-1)
#                 # Rotate the rotation_input by the rotation using quaternion multiplication
#                 q1 = rotation_input
#                 q2 = rotation
#                 w1, x1, y1, z1 = q1[:, :, 0], q1[:, :, 1], q1[:, :, 2], q1[:, :, 3]
#                 w2, x2, y2, z2 = q2[:, :, 0], q2[:, :, 1], q2[:, :, 2], q2[:, :, 3]
#                 w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#                 x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#                 y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#                 z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#                 rotation = torch.stack([w, x, y, z], dim=-1)

#             else:
#                 rotation = torch.tanh(rotation)
#                 rotation = rotation + latent[:, 6:10][:, None, :]

#         # Run activation
#         xyz_offset_norm = torch.sigmoid(xyz) * 2.0 - 1
#         if isinstance(voxel_size, torch.Tensor):
#             voxel_size = voxel_size[None, None, :]
#         xyz = xyz_offset_norm * voxel_size + xyz_voxel_center[:, None, :]
#         # print("scale before", scale.shape, scale.min(), scale.max())

#         if self.exp_scale:
#             scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
#             scale = torch.exp(scale) * voxel_size
#         else:
#             scale = torch.sigmoid(scale) * (voxel_size * self.max_scale)
#         rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)
#         # opacity = torch.clamp(opacity, max=MAX_BEFORE_SIGMOID, min=MIN_BEFORE_SIGMOID)
#         opacity = torch.sigmoid(opacity)
#         rgb = torch.sigmoid(rgb)

#         return {
#             "xyz_offset_norm": xyz_offset_norm,
#             "xyz": xyz,
#             "scale": scale,
#             "rotation": rotation,
#             "opacity": opacity,
#             "rgb": rgb,
#         }


class GSDecoder3D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_gs: int,
        skip_connect: bool,
        scale_activation: Literal["sigmoid", "exp", "softplus", "abs"] = "sigmoid",
        max_scale: float = 100.0,
        quat_rotation: bool = False,
        normal_zero_init: bool = False,
        modify_xyz_offsets: bool = True,
        offset_scaling: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.num_gs = num_gs
        self.skip_connect = skip_connect
        self.scale_activation = scale_activation
        self.max_scale = max_scale
        self.quat_rotation = quat_rotation
        self.normal_zero_init = normal_zero_init
        self.modify_xyz_offsets = modify_xyz_offsets
        self.offset_scaling = offset_scaling
        # self.out_dim = 3 + 3 + 4 + 1 + 3

        self.build_mlps()

    def build_mlps(self):
        self.mlp = MLP(
            in_dim=self.input_dim,
            layer_width=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation=nn.ReLU(),
            normal_zero_init=self.normal_zero_init,
        )

        self.xyz_head = nn.Linear(self.hidden_dim, 3 * self.num_gs)
        self.scale_head = nn.Linear(self.hidden_dim, 3 * self.num_gs)
        self.rotation_head = nn.Linear(self.hidden_dim, 4 * self.num_gs)
        self.opacity_head = nn.Linear(self.hidden_dim, 1 * self.num_gs)
        self.rgb_head = nn.Linear(self.hidden_dim, 3 * self.num_gs)

        if self.normal_zero_init:
            for head in [self.xyz_head, self.scale_head, self.rotation_head, self.opacity_head, self.rgb_head]:
                # nn.init.normal_(head.weight, mean=0.0, std=np.sqrt(2.0 / hidden_dim))
                nn.init.xavier_normal_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        xyz_voxel: torch.Tensor,
        transform: torch.Tensor,
        voxel_size: Union[float, torch.Tensor],
        override_offset_scaling: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.mlp(latent)
        xyz = self.xyz_head(out).reshape(-1, self.num_gs, 3)
        scale = self.scale_head(out).reshape(-1, self.num_gs, 3)
        rotation = self.rotation_head(out).reshape(-1, self.num_gs, 4)
        opacity = self.opacity_head(out).reshape(-1, self.num_gs, 1)
        rgb = self.rgb_head(out).reshape(-1, self.num_gs, 3)

        if self.skip_connect:
            xyz = torch.tanh(xyz)
            scale = torch.tanh(scale)
            opacity = torch.tanh(opacity)
            rgb = torch.tanh(rgb)

            if self.modify_xyz_offsets:
                xyz = xyz + latent[:, :3][:, None, :]
            else:
                xyz = latent[:, :3][:, None, :]
                xyz = xyz.expand(-1, self.num_gs, -1)
            scale = scale + latent[:, 3:6][:, None, :]
            opacity = opacity + latent[:, 10:11][:, None, :]
            rgb = rgb + latent[:, 11:14][:, None, :]

            if self.quat_rotation:
                raise NotImplementedError
            else:
                rotation = torch.tanh(rotation)
                rotation = rotation + latent[:, 6:10][:, None, :]

        xyz_offset_norm = torch.sigmoid(xyz) * 2.0 - 1
        if override_offset_scaling is None:
            xyz_offset_norm *= self.offset_scaling
        else:
            xyz_offset_norm *= override_offset_scaling

        xyz = xyz_voxel.float()[:, None, :] + xyz_offset_norm
        xyz = apply_transform(xyz, transform)

        if self.scale_activation == "exp":
            scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
            scale = torch.exp(scale) * voxel_size
        elif self.scale_activation == "softplus":
            scale = torch.nn.functional.softplus(scale) * voxel_size
        elif self.scale_activation == "abs":
            scale = torch.abs(scale) * voxel_size
        else:   # sigmoid
            scale = torch.sigmoid(scale) * (voxel_size * self.max_scale)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)

        # TODO: Whether to apply transformation on rotation
        opacity = torch.sigmoid(opacity)
        rgb = torch.sigmoid(rgb)

        return {
            "xyz_offset_norm": xyz_offset_norm,
            "xyz": xyz,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
            "rgb": rgb,
        }


class GSDecoder2D(GSDecoder3D):
    def build_mlps(self):
        self.mlp = MLP(
            in_dim=self.input_dim,
            layer_width=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation=nn.ReLU(),
            normal_zero_init=self.normal_zero_init,
        )

        self.xyz_head = nn.Linear(self.hidden_dim, 3 * self.num_gs)
        self.scale_head = nn.Linear(self.hidden_dim, 2 * self.num_gs)
        self.rotation_head = nn.Linear(self.hidden_dim, 4 * self.num_gs)
        self.opacity_head = nn.Linear(self.hidden_dim, 1 * self.num_gs)
        self.rgb_head = nn.Linear(self.hidden_dim, 3 * self.num_gs)

        if self.normal_zero_init:
            for head in [self.xyz_head, self.scale_head, self.rotation_head, self.opacity_head, self.rgb_head]:
                # nn.init.normal_(head.weight, mean=0.0, std=np.sqrt(2.0 / hidden_dim))
                nn.init.xavier_normal_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        xyz_voxel: torch.Tensor,
        transform: torch.Tensor,
        voxel_size: Union[float, torch.Tensor],
        override_offset_scaling: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.mlp(latent)
        xyz = self.xyz_head(out).reshape(-1, self.num_gs, 3)
        scale = self.scale_head(out).reshape(-1, self.num_gs, 2)
        rotation = self.rotation_head(out).reshape(-1, self.num_gs, 4)
        opacity = self.opacity_head(out).reshape(-1, self.num_gs, 1)
        rgb = self.rgb_head(out).reshape(-1, self.num_gs, 3)

        if self.skip_connect:
            xyz = torch.tanh(xyz)
            scale = torch.tanh(scale)
            opacity = torch.tanh(opacity)
            rgb = torch.tanh(rgb)

            if self.modify_xyz_offsets:
                xyz = xyz + latent[:, :3][:, None, :]
            else:
                xyz = latent[:, :3][:, None, :]
                xyz = xyz.expand(-1, self.num_gs, -1)
            # scale = scale + latent[:, 3:6][:, None, :]
            # opacity = opacity + latent[:, 10:11][:, None, :]
            # rgb = rgb + latent[:, 11:14][:, None, :]
            scale = scale + latent[:, 3:5][:, None, :]
            opacity = opacity + latent[:, 9:10][:, None, :]
            rgb = rgb + latent[:, 10:13][:, None, :]

            if self.quat_rotation:
                raise NotImplementedError
            else:
                rotation = torch.tanh(rotation)
                rotation = rotation + latent[:, 5:9][:, None, :]

        xyz_offset_norm = torch.sigmoid(xyz) * 2.0 - 1
        if override_offset_scaling is None:
            xyz_offset_norm *= self.offset_scaling
        else:
            xyz_offset_norm *= override_offset_scaling

        xyz = xyz_voxel.float()[:, None, :] + xyz_offset_norm
        xyz = apply_transform(xyz, transform)

        if self.scale_activation == "exp":
            scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
            scale = torch.exp(scale) * voxel_size
        elif self.scale_activation == "softplus":
            scale = torch.nn.functional.softplus(scale) * voxel_size
        elif self.scale_activation == "abs":
            scale = torch.abs(scale) * voxel_size
        else:   # sigmoid
            scale = torch.sigmoid(scale) * (voxel_size * self.max_scale)

        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)
        # TODO: Whether to apply transformation on rotation
        opacity = torch.sigmoid(opacity)
        rgb = torch.sigmoid(rgb)

        return {
            "xyz_offset_norm": xyz_offset_norm,
            "xyz": xyz,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
            "rgb": rgb,
        }


class GSIdentityDecoder3D(nn.Module):
    def __init__(
        self,
        max_scale: float = 100.0,
        scale_activation: Literal["sigmoid", "exp", "softplus", "abs"] = "sigmoid",
        offset_scaling: float = 1.0,
    ):
        super().__init__()
        self.max_scale = max_scale
        self.scale_activation = scale_activation
        self.offset_scaling = offset_scaling

    def forward(
        self,
        latent: torch.Tensor,
        xyz_voxel: torch.Tensor,
        transform: torch.Tensor,
        voxel_size: Union[float, torch.Tensor],
        override_offset_scaling: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        xyz = latent[:, :3]
        scale = latent[:, 3:6]
        rotation = latent[:, 6:10]
        opacity = latent[:, 10:11]
        rgb = latent[:, 11:14]

        xyz_offset_norm = torch.sigmoid(xyz) * 2.0 - 1
        if override_offset_scaling is None:
            xyz_offset_norm *= self.offset_scaling
        else:
            xyz_offset_norm *= override_offset_scaling

        # print(f"xyz_offset_norm: {xyz_offset_norm.min()}, {xyz_offset_norm.max()}, {xyz_offset_norm.mean()} {xyz_offset_norm.std()}")
        xyz = xyz_voxel.float() + xyz_offset_norm
        # print("voxel_to_world", transform)
        xyz = apply_transform(xyz, transform)

        if self.scale_activation == "exp":
            scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
            scale = torch.exp(scale) * voxel_size
        elif self.scale_activation == "softplus":
            scale = torch.nn.functional.softplus(scale) * voxel_size
        elif self.scale_activation == "abs":
            scale = torch.abs(scale) * voxel_size
        else:   # sigmoid
            scale = torch.sigmoid(scale) * (voxel_size * self.max_scale)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)

        # TODO: Whether to apply transformation on rotation
        opacity = torch.sigmoid(opacity)
        rgb = torch.sigmoid(rgb)

        return {
            "xyz_offset_norm": xyz_offset_norm.unsqueeze(1),
            "xyz": xyz.unsqueeze(1),
            "scale": scale.unsqueeze(1),
            "rotation": rotation.unsqueeze(1),
            "opacity": opacity.unsqueeze(1),
            "rgb": rgb.unsqueeze(1),
        }


class GSIdentityDecoder2D(GSIdentityDecoder3D):
    def forward(
        self,
        latent: torch.Tensor,
        xyz_voxel: torch.Tensor,
        transform: torch.Tensor,
        voxel_size: Union[float, torch.Tensor],
        override_offset_scaling: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Use only the first 2 dimensions for scale
        """
        xyz = latent[:, :3]
        scale = latent[:, 3:5]
        rotation = latent[:, 5:9]
        opacity = latent[:, 9:10]
        rgb = latent[:, 10:13]

        xyz_offset_norm = torch.sigmoid(xyz) * 2.0 - 1
        if override_offset_scaling is None:
            xyz_offset_norm *= self.offset_scaling
        else:
            xyz_offset_norm *= override_offset_scaling

        # print(f"xyz_offset_norm: {xyz_offset_norm.min()}, {xyz_offset_norm.max()}, {xyz_offset_norm.mean()} {xyz_offset_norm.std()}")
        xyz = xyz_voxel.float() + xyz_offset_norm
        # print("voxel_to_world", transform)
        xyz = apply_transform(xyz, transform)

        if self.scale_activation == "exp":
            scale = torch.clamp(scale, min=MIN_LOG_SCALE, max=MAX_LOG_SCALE)
            scale = torch.exp(scale) * voxel_size
        elif self.scale_activation == "softplus":
            scale = torch.nn.functional.softplus(scale) * voxel_size
        elif self.scale_activation == "abs":
            scale = torch.abs(scale) * voxel_size
        else:   # sigmoid
            scale = torch.sigmoid(scale) * (voxel_size * self.max_scale)
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)

        # TODO: Whether to apply transformation on rotation
        opacity = torch.sigmoid(opacity)
        rgb = torch.sigmoid(rgb)

        return {
            "xyz_offset_norm": xyz_offset_norm.unsqueeze(1),
            "xyz": xyz.unsqueeze(1),
            "scale": scale.unsqueeze(1),
            "rotation": rotation.unsqueeze(1),
            "opacity": opacity.unsqueeze(1),
            "rgb": rgb.unsqueeze(1),
        }
