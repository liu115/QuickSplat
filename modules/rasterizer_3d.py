from typing import Optional, Literal, Tuple, Dict
from dataclasses import dataclass
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
try:
    from gsplat import rasterization as gsplat_rasterization
except ImportError:
    print("GSplat is not installed")


@dataclass
class Camera:
    fov_x: float
    fov_y: float
    height: int
    width: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image: torch.Tensor
    depth: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: Dict[str, float]):
        return Camera(
            fov_x=d["fov_x"],
            fov_y=d["fov_y"],
            height=d["height"],
            width=d["width"],
        )


class Rasterizer(torch.nn.Module):
    def __init__(
        self,
        config,
        bg_color: torch.Tensor
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

    def render_depth(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        scale_modifier: float = 1.0,
    ):
        # Implement depth rendering using the depth of the gaussians as color

        xyz = gaussians.get_xyz
        pseudo_rgb = torch.ones_like(xyz)

        world_to_camera = camera.world_view_transform.T
        xyz_cam = torch.matmul(world_to_camera[:3, :3], xyz.T) + world_to_camera[:3, 3:4]

        gs_depth = xyz_cam[2]

        pseudo_rgb[:, 0] = gs_depth

        means3D = gaussians.get_xyz
        means2D = torch.zeros_like(means3D)
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        bg_color = torch.zeros_like(self.bg_color)      # [0, 0, 0]
        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, gaussians.active_sh_degree)
        render_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=pseudo_rgb,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        depth = render_image[:1]
        alpha = render_image[1:2]

        depth /= alpha.clamp(min=1e-8)
        depth = torch.nan_to_num(depth, 0, 0)
        return depth

    def forward(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        override_bg_color: Optional[torch.Tensor] = None,
        override_color: Optional[torch.Tensor] = None,
        scale_modifier: float = 1.0,
    ):
        if override_bg_color is not None:
            bg_color = override_bg_color
        else:
            bg_color = self.bg_color

        assert len(bg_color.shape) == 1
        assert bg_color.shape[0] == 3
        if override_color is not None:
            assert override_color.shape[0] == gaussians.get_xyz.shape[0]
        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, gaussians.active_sh_degree)

        screenspace_points = torch.zeros_like(
            gaussians.get_xyz, requires_grad=True
        )
        screenspace_points.retain_grad()
        means3D = gaussians.get_xyz
        means2D = screenspace_points
        opacity = gaussians.get_opacity
        shs = gaussians.get_features
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        # rendered_image, radii, n_contrib, alphas, indices, depths = rasterizer(
        # print(means2D.shape, means2D.device)
        # print(means3D.shape, means3D.device)
        # print(shs.shape, shs.device)
        # print(opacity.shape, opacity.device)
        # print(scales.shape, scales.device)
        # print(rotations.shape, rotations.device)
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


class ScaffoldRasterizer(Rasterizer):
    def forward(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera: Camera,
        active_sh_degree: int,
        override_bg_color: Optional[torch.Tensor] = None,
        override_color: Optional[torch.Tensor] = None,
        scale_modifier: float = 1.0,
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

        # means3D = gaussians.get_xyz
        # opacity = gaussians.get_opacity
        # shs = gaussians.get_features
        # scales = gaussians.get_scaling
        # rotations = gaussians.get_rotation
        # means3D, scales, rotations, opacity, shs = gaussians.get_all_attributes()

        means3D = gaussian_params["xyz"]
        scales = gaussian_params["scale"]
        rotations = gaussian_params["rotation"]
        opacity = gaussian_params["opacity"]
        if "shs" in gaussian_params:
            shs = gaussian_params["shs"]
            rgb = None
        else:
            shs = None
            rgb = gaussian_params["rgb"]

        screenspace_points = torch.zeros_like(
            means3D, requires_grad=True
        )
        screenspace_points.retain_grad()
        means2D = screenspace_points

        # rendered_image, radii, n_contrib, alphas, indices, depths = rasterizer(
        # print(means2D.shape, means2D.device)
        # print(means3D.shape, means3D.device)
        # print(shs.shape, shs.device)
        # print(opacity.shape, opacity.device)
        # print(scales.shape, scales.device)
        # print(rotations.shape, rotations.device)
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def render_depth(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera: Camera,
        scale_modifier: float = 1.0,
    ):
        # Implement depth rendering using the depth of the gaussians as color
        xyz = gaussian_params["xyz"]
        pseudo_rgb = torch.ones_like(xyz)

        world_to_camera = camera.world_view_transform.T
        xyz_cam = torch.matmul(world_to_camera[:3, :3], xyz.T) + world_to_camera[:3, 3:4]
        gs_depth = xyz_cam[2]

        pseudo_rgb[:, 0] = gs_depth

        means3D = gaussian_params["xyz"]
        means2D = torch.zeros_like(means3D)
        opacity = gaussian_params["opacity"]
        scales = gaussian_params["scale"]
        rotations = gaussian_params["rotation"]
        if "shs" in gaussian_params:
            shs = gaussian_params["shs"]
            rgb = None
        else:
            shs = None
            rgb = gaussian_params["rgb"]

        bg_color = torch.zeros_like(self.bg_color)      # [0, 0, 0]
        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, 3)
        render_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        depth = render_image[:1]
        alpha = render_image[1:2]

        depth /= alpha.clamp(min=1e-8)
        depth = torch.nan_to_num(depth, 0, 0)
        return depth


class SimpleRasterizer(torch.nn.Module):
    def __init__(
        self,
        config,
        bg_color: torch.Tensor,
        random_bg_color: bool = False,
    ):
        super().__init__()
        self.scaling_modifier = 1.0
        self.bg_color = bg_color
        self.random_bg_color = random_bg_color
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
        camera: Camera,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor],
        shs: Optional[torch.Tensor],
        opacity: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        scale_modifier: float = 1.0,
        active_sh_degree: int = 3,
        override_bg_color: Optional[torch.Tensor] = None,
    ):
        if override_bg_color is not None:
            bg_color = override_bg_color
        elif self.random_bg_color:
            bg_color = torch.rand(3).to(self.bg_color.device)
        else:
            bg_color = self.bg_color
        assert len(bg_color.shape) == 1
        assert bg_color.shape[0] == 3
        assert rgb is not None or shs is not None
        assert rgb is None or shs is None

        rasterizer = self.get_rasterizer(camera, bg_color, scale_modifier, active_sh_degree)

        screenspace_points = torch.zeros_like(xyz)
        means3D = xyz
        means2D = screenspace_points
        opacity = opacity

        assert xyz.is_contiguous()
        assert opacity.is_contiguous()
        assert scales.is_contiguous()
        assert rotations.is_contiguous()
        if shs is not None:
            assert shs.is_contiguous()
        if rgb is not None:
            assert rgb.is_contiguous()

        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return {
            "render": rendered_image,
            "radii": radii,
        }
