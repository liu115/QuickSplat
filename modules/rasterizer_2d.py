from typing import Optional, Literal, Tuple, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from modules.rasterizer_3d import Camera


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.width, view.height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


class Scaffold2DGSRasterizer(nn.Module):
    def __init__(
        self,
        config,
        bg_color: torch.Tensor,
    ):
        super().__init__()
        self.scaling_modifier = 1.0
        self.bg_color = bg_color
        self.config = config

    def get_rasterizer(
        self,
        camera: Camera,
        bg_color: torch.Tensor,
        scale_modifier: float,
        active_sh_degree: int,
    ):
        tan_fov_x = math.tan(camera.fov_x / 2)
        tan_fov_y = math.tan(camera.fov_y / 2)

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.height,
            image_width=camera.width,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=bg_color,
            scale_modifier=scale_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

    def forward(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera: Camera,
        active_sh_degree: int,
        override_bg_color: Optional[torch.Tensor] = None,
        override_color: Optional[torch.Tensor] = None,
        scale_modifier: float = 1.0,
        depth_source: Literal["median", "expected"] = "expected",
    ):
        if override_bg_color is not None:
            bg_color = override_bg_color
        else:
            bg_color = self.bg_color

        num_gaussians = gaussian_params["xyz"].shape[0]

        assert len(bg_color.shape) == 1
        assert bg_color.shape[0] == 3
        if override_color is not None:
            assert override_color.shape[0] == num_gaussians
        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, active_sh_degree)

        means3D = gaussian_params["xyz"]
        if means3D.ndim != 2:
            means3D = means3D.view(-1, 3)
        means3D = means3D.contiguous()

        scales = gaussian_params["scale"]
        if scales.ndim != 2:
            scales = scales.view(-1, scales.shape[-1])
        if scales.shape[1] == 3:
            # NOTE: Use only two dimensions for scale in 2DGS. drop the last dimension if it exists
            scales = scales[:, :2]
        scales = scales.contiguous()

        rotations = gaussian_params["rotation"]
        if rotations.ndim != 2:
            rotations = rotations.view(-1, rotations.shape[-1])
        rotations = rotations.contiguous()

        opacity = gaussian_params["opacity"]
        if opacity.ndim != 2:
            opacity = opacity.view(-1, 1)
        opacity = opacity.contiguous()

        if "shs" in gaussian_params:
            shs = gaussian_params["shs"]
            if shs.ndim != 3:
                shs = shs.view(-1, shs.shape[-2], shs.shape[-1])
            shs = shs.contiguous()
            rgb = None
        else:
            shs = None
            rgb = gaussian_params["rgb"]
            if rgb.ndim != 2:
                rgb = rgb.view(-1, 3)
            rgb = rgb.contiguous()

        means2D = torch.zeros_like(means3D, requires_grad=True)
        means2D.retain_grad()

        rendered_image, radii, other_maps = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        render_alpha = other_maps[1:2]

        render_normal = other_maps[2:5]
        render_normal = render_normal.permute(1, 2, 0) @ (camera.world_view_transform[:3, :3].T)
        render_normal = render_normal.permute(2, 0, 1)

        # get median depth map
        render_depth_median = other_maps[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = other_maps[0:1]
        render_depth_expected = (render_depth_expected / torch.clamp_min(render_alpha, 1e-6))
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = other_maps[6:7]

        if depth_source == "median":
            render_depth = render_depth_median
        elif depth_source == "expected":
            render_depth = render_depth_expected
        else:
            raise ValueError(f"Unknown depth source {depth_source}")

        surf_normal = depth_to_normal(camera, render_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

        return {
            "render": rendered_image,
            "depth": render_depth,
            "depth_median": render_depth_median,
            "depth_expected": render_depth_expected,
            "alpha": render_alpha,
            "normal": render_normal,
            "normal_from_depth": surf_normal,
            "distortion": render_dist,
            "radii": radii,
            "viewspace_points": means2D,
            "visibility_filter": radii > 0,
        }


@dataclass
class GaussianModelAbstract:
    get_xyz: torch.Tensor
    get_opacity: torch.Tensor
    get_features: torch.Tensor
    get_scaling: torch.Tensor
    get_rotation: torch.Tensor
    active_sh_degree: int


class Full2DGSRasterizer(Scaffold2DGSRasterizer):
    def forward(
        self,
        gaussians: GaussianModelAbstract,
        camera: Camera,
        override_bg_color: Optional[torch.Tensor] = None,
        override_color: Optional[torch.Tensor] = None,
        scale_modifier: float = 1.0,
        depth_source: Literal["median", "expected"] = "expected",
    ):
        if override_bg_color is not None:
            bg_color = override_bg_color
        else:
            bg_color = self.bg_color

        # The device of bg_color is the same as the device of gaussians
        assert len(bg_color.shape) == 1
        assert bg_color.shape[0] == 3
        assert bg_color.device == gaussians.get_xyz.device

        if override_color is not None:
            assert override_color.shape[0] == gaussians.get_xyz.shape[0]
        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, gaussians.active_sh_degree)

        screenspace_points = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
        screenspace_points.retain_grad()

        means3D = gaussians.get_xyz.contiguous()
        opacity = gaussians.get_opacity.contiguous()
        shs = gaussians.get_features.contiguous()
        scales = gaussians.get_scaling.contiguous()
        rotations = gaussians.get_rotation.contiguous()

        rendered_image, radii, other_maps = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        render_alpha = other_maps[1:2]

        render_normal = other_maps[2:5]
        render_normal = render_normal.permute(1, 2, 0) @ (camera.world_view_transform[:3, :3].T)
        render_normal = render_normal.permute(2, 0, 1)

        # get median depth map
        render_depth_median = other_maps[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = other_maps[0:1]
        render_depth_expected = (render_depth_expected / torch.clamp_min(render_alpha, 1e-6))
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = other_maps[6:7]

        if depth_source == "median":
            render_depth = render_depth_median
        elif depth_source == "expected":
            render_depth = render_depth_expected
        else:
            raise ValueError(f"Unknown depth source {depth_source}")

        surf_normal = depth_to_normal(camera, render_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach()

        return {
            "viewspace_points": screenspace_points,
            "render": rendered_image,
            "depth": render_depth,
            "depth_median": render_depth_median,
            "depth_expected": render_depth_expected,
            "alpha": render_alpha,
            "normal": render_normal,
            "normal_from_depth": surf_normal,
            "distortion": render_dist,
            "radii": radii,
            "visibility_filter": radii > 0,
        }
