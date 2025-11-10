from typing import Optional, Literal, Tuple, Dict
import math
import os

try:
    from simple_knn.distCUDA2 import distCUDA2
except ImportError:
    # Only create_from_points will be affected (not used by default)
    pass

import torch
import torch.nn as nn
import pytorch3d.transforms as transforms3d

from utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    RGB2SH,
    eval_sh,
    SH2RGB,
)


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


class Gaussian2DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.active_sh_degree = config.init_sh_degree
        self.max_sh_degree = 3
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_rgb(self):
        # Convert SH to RGB (using the first sh degree)
        return SH2RGB(self.get_features[:, 0, :3])

    @property
    def num_points(self):
        return self._xyz.shape[0]

    @staticmethod
    def create_from_gaussians(
        config,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
    ):
        num_points = xyz.shape[0]
        model = Gaussian2DModel(config)
        model.spatial_lr_scale = 1.0
        fused_color = RGB2SH(rgb)
        features = torch.zeros((num_points, 3, (model.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        # N, 2
        scales = torch.log(scales)
        # N, 1
        opacities = inverse_sigmoid(opacities)
        model._xyz = nn.Parameter(xyz, requires_grad=True)
        model._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True)
        model._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous(), requires_grad=True)

        model._scaling = nn.Parameter(scales, requires_grad=True)
        model._rotation = nn.Parameter(rotations, requires_grad=True)
        model._opacity = nn.Parameter(opacities, requires_grad=True)
        model.max_radii2D = torch.zeros(num_points, device="cuda")
        return model

    @staticmethod
    def create_from_points(
        config,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        spatial_lr_scale: float,
        knn_dist2: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
    ):
        num_points = xyz.shape[0]
        print(f"Creating 2DGS with {num_points} points")
        assert xyz.shape[0] == rgb.shape[0]

        model = Gaussian2DModel(config)
        model.spatial_lr_scale = spatial_lr_scale

        fused_color = RGB2SH(rgb)
        features = torch.zeros((num_points, 3, (model.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0     # Already zero

        if knn_dist2 is None:
            dist2 = torch.clamp_min(distCUDA2(xyz.cuda()), 0.0000001)
        else:
            dist2 = torch.clamp_min(knn_dist2, 0.0000001).cuda()
            # TODO: Clamp max

        # KEY DIFFERENCE: Scale is 2D
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)

        # KEY DIFFERENCE: Random rotation initialization
        if normal is None:
            rots = torch.rand((num_points, 4), device="cuda")
        else:
            # Use the normal to initialize the rotation
            # TODO: Test this
            rot_mat = compute_rotation_from_normal(normal)
            rot_mat = rot_mat.view(-1, 3, 3)
            rots = transforms3d.matrix_to_quaternion(rot_mat)

        opacities = inverse_sigmoid(config.init_opacity * torch.ones((num_points, 1), dtype=torch.float32, device="cuda"))

        model._xyz = nn.Parameter(xyz, requires_grad=True)
        model._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True)
        model._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous(), requires_grad=True)

        model._scaling = nn.Parameter(scales, requires_grad=True)
        model._rotation = nn.Parameter(rots, requires_grad=True)
        model._opacity = nn.Parameter(opacities, requires_grad=True)
        model.max_radii2D = torch.zeros(num_points, device="cuda")
        return model

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device="cuda")
        self.denom = torch.zeros((self.num_points, 1), device="cuda")
        optim_config = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_rest_lr, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(optim_config, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def upate_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def save_ckpt(self, path, step=0):
        xyz = self._xyz.detach().cpu()
        f_dc = self._features_dc.detach().cpu()
        f_rest = self._features_rest.detach().cpu()
        opacity = self._opacity.detach().cpu()
        scale = self._scaling.detach().cpu()
        rotation = self._rotation.detach().cpu()

        ckpt = {
            "optimizer": self.optimizer.state_dict(),
            "xyz": xyz,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacity,
            "scaling": scale,
            "rotation": rotation,
            "step": step,
        }

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)

    def load_ckpt(self, config, path, load_optimizer: bool = True) -> int:

        ckpt = torch.load(path, map_location="cpu")
        self._xyz = nn.Parameter(ckpt["xyz"].cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(ckpt["f_dc"].cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(ckpt["f_rest"].cuda().requires_grad_(True))
        self._opacity = nn.Parameter(ckpt["opacity"].cuda().requires_grad_(True))
        self._scaling = nn.Parameter(ckpt["scaling"].cuda().requires_grad_(True))
        self._rotation = nn.Parameter(ckpt["rotation"].cuda().requires_grad_(True))

        self.training_setup(config)
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.gaussian_weights_acc = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        print(f"Loaded checkpoint from {path}. #Points: {self.get_xyz.shape[0]}. #iterations: {ckpt['step']}")
        return ckpt["step"]

    def save_ply(self, path):
        # TODO: Save the gaussian as .ply file for visualization
        pass

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size=None, max_scale_3d=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        self.prune(min_opacity, extent, max_screen_size, max_scale_3d)

    def prune(self, min_opacity, extent, max_screen_size, max_scale_3d):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size is not None and max_screen_size > 0:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # Update: Added 3D scale pruning
        if max_scale_3d is not None and max_scale_3d > 0:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale_3d
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
