from typing import Any, Union, Optional, List, Tuple, Dict, Literal
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import MinkowskiEngine as ME
from pytorch3d import ops
from pytorch3d import transforms as transforms3d

from trainers.quicksplat_base_trainer import QuickSplatTrainer
from models.optimizer import UNetOptimizer
from models.initializer import Initializer18
from models.densifier import Densifier
from models.unet_base import ResUNetConfig
from models.scaffold_gs import ScaffoldGSFull, inverse_sigmoid, inverse_softplus
from modules.rasterizer_3d import ScaffoldRasterizer, Camera
from modules.rasterizer_2d import Scaffold2DGSRasterizer

from utils.pose import apply_transform, apply_rotation
from utils.optimizer import Optimizer
from utils.rich_utils import CONSOLE
from utils.sparse import xyz_list_to_bxyz, chamfer_dist, chamfer_dist_with_crop
from utils.fusion import MeshExtractor


class Phase2Trainer(QuickSplatTrainer):

    def setup_densifier(self):
        if self.config.MODEL.DENSIFIER.enable:
            self.densifier = Densifier(
                config=self.config,
                in_dim=self.config.MODEL.SCAFFOLD.hidden_dim,
                out_dim=self.config.MODEL.SCAFFOLD.hidden_dim,
                is_2dgs=self.config.MODEL.gaussian_type == "2d",
                occupancy_as_opacity=self.config.MODEL.DENSIFIER.occupancy_as_opacity,
            ).to(self.device)

            if self.world_size > 1:
                self.densifier = DDP(self.densifier, device_ids=[self.local_rank], find_unused_parameters=True)

        self.prune = ME.MinkowskiPruning()

    def setup_modules(self):
        super().setup_modules()

        self.bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)   # White
        if self.config.MODEL.gaussian_type == "3d":
            self.rasterizer = ScaffoldRasterizer(
                self.config.MODEL.GSPLAT,
                self.bg_color,
            )
        elif self.config.MODEL.gaussian_type == "2d":
            self.rasterizer = Scaffold2DGSRasterizer(
                self.config.MODEL.GSPLAT,
                self.bg_color,
            )
        else:
            raise ValueError(f"Unknown gaussian type {self.config.MODEL.gaussian_type}")

        if self.config.MODEL.input_type == "colmap+completion":
            if self.config.MODEL.gaussian_type == "3d":
                out_channels = 3 + 3 + 4    # scale (3) + rgb (3) + rotation (4)
            else:
                out_channels = 2 + 3 + 4    # scale (2) + rgb (3) + rotation (4)
            self.initializer = Initializer18(
                in_channels=3,
                out_channels=out_channels,
                config=ResUNetConfig(),
                use_time_emb=False,
                dense_bottleneck=True,
                num_dense_blocks=self.config.MODEL.INIT.num_dense_blocks,
            ).to(self.device).eval()

            # Load the initializer checkpoint
            ckpt_path = Path(self.config.TRAIN.ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist"
            # If the checkpoint is a directory, load the latest checkpoint
            if ckpt_path.is_dir():
                ckpt_path = sorted(ckpt_path.glob("*.ckpt"))[-1]
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.initializer.load_state_dict(ckpt["model"])
            CONSOLE.print(f"Loaded Initializer checkpoint from {ckpt_path}")

    def get_model(self) -> nn.Module:
        self.setup_decoders()
        self.setup_densifier()
        return UNetOptimizer(
            in_dim=self.config.MODEL.SCAFFOLD.hidden_dim * 2,
            out_dim=self.config.MODEL.SCAFFOLD.hidden_dim,
            backbone=self.config.MODEL.OPT.backbone,
            output_norm=self.config.MODEL.OPT.output_norm,
            grad_residual=self.config.MODEL.OPT.grad_residual,
        )

    def setup_optimizers(self):
        # param_dict = self.model.get_param_groups()
        param_dict = {
            "model": self.model.parameters(),
            "decoder": self.scaffold_decoder.parameters(),
        }
        if self.config.MODEL.DENSIFIER.enable:
            param_dict["densifier"] = self.densifier.parameters()
        self.optimizer = Optimizer(param_dict, self.config.OPTIMIZER)

    def get_all_model_state_dict(self):
        all_state_dict = {}
        all_state_dict["model"] = self.model.state_dict()

        if self.config.MODEL.DENSIFIER.enable:
            if self.world_size > 1:
                all_state_dict["densifier"] = self.densifier.module.state_dict()
            else:
                all_state_dict["densifier"] = self.densifier.state_dict()
        if self.config.MODEL.decoder_type == "scaffold":
            if self.world_size > 1:
                all_state_dict["decoder"] = self.scaffold_decoder.module.state_dict()
            else:
                all_state_dict["decoder"] = self.scaffold_decoder.state_dict()
        return all_state_dict

    @torch.no_grad()
    def init_scaffold_train_batch(
        self,
        xyz_list: List[torch.Tensor],
        rgb_list: List[torch.Tensor],
        xyz_voxel_list: List[torch.Tensor],
        transform_list: List[torch.Tensor],     # voxel_to_world transform
        bbox_list: List[torch.Tensor],
        bbox_voxel_list: Optional[List[torch.Tensor]] = None,
        test_mode: bool = False,
    ) -> List[ScaffoldGSFull]:
        batch_size = len(xyz_list)
        scaffold_list = []

        bxyz_input, _ = xyz_list_to_bxyz(xyz_voxel_list)
        rgb_input = torch.cat(rgb_list, dim=0)
        sparse_tensor_input = ME.SparseTensor(features=rgb_input, coordinates=bxyz_input)
        initializer_outputs = self.initializer(
            sparse_tensor_input,
            gt_coords=bxyz_input,       # Doesn't matter for evaluation
            # max_samples_second_last=self.config.DATASET.max_num_points * self.config.TRAIN.batch_size // 8,
            # verbose=self.config.debug,
        )
        output_sparse = initializer_outputs["out"]
        occ_logits_last = initializer_outputs["last_prob"]

        xyz_voxel_init, params_init = output_sparse.decomposed_coordinates_and_features
        for i in range(batch_size):
            valid_mask = (xyz_voxel_init[i] >= bbox_voxel_list[i][0, :]).all(dim=1) & (xyz_voxel_init[i] <= bbox_voxel_list[i][1, :]).all(dim=1)
            xyz_voxel_init[i] = xyz_voxel_init[i][valid_mask]
            params_init[i] = params_init[i][valid_mask]
            occ = occ_logits_last.features_at(batch_index=i)[valid_mask]

            if not test_mode and xyz_voxel_init[i].shape[0] > self.config.DATASET.max_num_points:
                indices = torch.randperm(xyz_voxel_init[i].shape[0])[:self.config.DATASET.max_num_points]
                xyz_voxel_init[i] = xyz_voxel_init[i][indices]
                params_init[i] = params_init[i][indices]
                occ = occ[indices]
            elif test_mode and xyz_voxel_init[i].shape[0] > self.config.DATASET.eval_max_num_points:
                indices = torch.randperm(xyz_voxel_init[i].shape[0])[:self.config.DATASET.eval_max_num_points]
                xyz_voxel_init[i] = xyz_voxel_init[i][indices]
                params_init[i] = params_init[i][indices]
                occ = occ[indices]

            if self.config.MODEL.gaussian_type == "3d":
                latent_init = self._build_init_latents(
                    xyz_voxel_orig=xyz_voxel_list[i],
                    rgb_orig=rgb_list[i],
                    xyz_voxel_init=xyz_voxel_init[i],
                    scale=params_init[i][:, 0:3],
                    opacity=occ,
                    rgb=params_init[i][:, 3:6],
                    rotation=params_init[i][:, 6:],
                    voxel_to_world=transform_list[i],
                )
            else:
                latent_init = self._build_init_latents(
                    xyz_voxel_orig=xyz_voxel_list[i],
                    rgb_orig=rgb_list[i],
                    xyz_voxel_init=xyz_voxel_init[i],
                    scale=params_init[i][:, 0:2],
                    opacity=occ,
                    rgb=params_init[i][:, 2:5],
                    rotation=params_init[i][:, 5:],
                    voxel_to_world=transform_list[i],
                )

            scaffold = ScaffoldGSFull.create_from_latent(
                self.config.MODEL,
                xyz_voxel=xyz_voxel_init[i],
                latent=latent_init,
                bbox=bbox_list[i],
                voxel_to_world_transform=transform_list[i],
                voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
                spatial_lr_scale=1.0,
            )
            scaffold_list.append(scaffold)

        return scaffold_list

    # def init_scaffold_test(self, scene_id: str, test_mode: bool = False) -> ScaffoldGSFull:
    #     assert test_mode, "Test mode must be True in init_scaffold_test"

    #     # By default, don't use normal unless specified
    #     normal = None
    #     bbox_voxel = None
    #     if self.config.MODEL.input_type == "colmap" or self.config.MODEL.input_type == "colmap+completion":
    #         crop_points = True
    #         # crop_points = False
    #         # if self.config.MODEL.input_type == "colmap+completion":
    #         #     # To help the completion
    #         #     crop_points = True

    #         # xyz, rgb = self.val_dataset.load_colmap_points(scene_id, crop_points=crop_points)
    #         # xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel = voxelize(
    #         #     xyz,
    #         #     rgb,
    #         #     voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #         # )
    #         xyz, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel = self.val_dataset.load_voxelized_colmap_points(
    #             scene_id,
    #             voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #             crop_points=crop_points,
    #         )
    #         bbox_voxel = torch.from_numpy(bbox_voxel).int().to(self.device)

    #     elif self.config.MODEL.input_type == "mesh":
    #         if self.config.MODEL.init_type == "xyz+rgb+normal":
    #             xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, normal = self.val_dataset.load_voxelized_mesh_points(
    #                 scene_id,
    #                 voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #                 load_normal=True,
    #             )
    #             # xyz, rgb, normal = self.val_dataset.load_point_with_normal(scene_id)

    #             # xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, normal = voxelize(
    #             #     xyz,
    #             #     rgb,
    #             #     voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #             #     xyz_world=normal,
    #             # )
    #             normal = torch.from_numpy(normal).float().to(self.device)

    #         elif self.config.MODEL.init_type == "xyz+rgb":
    #             # xyz, rgb = self.val_dataset.load_mesh_points(scene_id)
    #             # xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel = voxelize(
    #             #     xyz,
    #             #     rgb,
    #             #     voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #             # )
    #             xyz, rgb, xyz_voxel, xyz_offset, bbox, world_to_voxel, _ = self.val_dataset.load_voxelized_mesh_points(
    #                 scene_id,
    #                 voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #                 load_normal=False,
    #             )

    #     xyz = torch.from_numpy(xyz).float().to(self.device)
    #     rgb = torch.from_numpy(rgb).float().to(self.device)
    #     xyz_voxel = torch.from_numpy(xyz_voxel).float().to(self.device)
    #     xyz_offset = torch.from_numpy(xyz_offset).float().to(self.device)
    #     voxel_to_world = torch.from_numpy(world_to_voxel).float().to(self.device).inverse()
    #     bbox = torch.from_numpy(bbox).float().to(self.device)

    #     return self.init_scaffold_train(
    #         xyz,
    #         rgb,
    #         xyz_voxel,
    #         xyz_offset,
    #         voxel_to_world,
    #         bbox,
    #         bbox_voxel=bbox_voxel,
    #         normal=normal,
    #         test_mode=test_mode,
    #     )

    # def init_scaffold_train(
    #     self,
    #     xyz: torch.Tensor,
    #     rgb: torch.Tensor,
    #     xyz_voxel: torch.Tensor,
    #     xyz_offset: torch.Tensor,
    #     transform: torch.Tensor,    # From voxel to world
    #     bbox: torch.Tensor,
    #     bbox_voxel: Optional[torch.Tensor] = None,
    #     normal: Optional[torch.Tensor] = None,
    #     test_mode: bool = False,
    # ) -> ScaffoldGSFull:
    #     seed = None
    #     if self.config.MODEL.SCAFFOLD.fix_init or test_mode:
    #         seed = self.config.MODEL.SCAFFOLD.seed

    #     if self.config.MODEL.input_type == "colmap+completion":
    #         # Use the initializer to densify the points
    #         xyz_voxel, xyz, rgb, normal = self._densify(xyz_voxel, xyz, rgb, transform, bbox, bbox_voxel=bbox_voxel, test_mode=test_mode)
    #         # After completion, the offsets are lost
    #         xyz_offset = torch.zeros_like(xyz)
    #         # TODO: Check this
    #         if "normal" not in self.config.MODEL.init_type:
    #             normal = None

    #     return ScaffoldGSFull.create_from_voxels2(
    #         self.config.MODEL,
    #         hidden_dim=self.config.MODEL.SCAFFOLD.hidden_dim,
    #         voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
    #         xyz=xyz,
    #         rgb=rgb,
    #         xyz_voxel=xyz_voxel,
    #         xyz_offset=xyz_offset,
    #         transform=transform,
    #         bbox=bbox,
    #         zero_latent=self.config.MODEL.SCAFFOLD.zero_latent,
    #         spatial_lr_scale=1.0,
    #         normal=normal,
    #         seed=seed,
    #         unit_scale=self.config.MODEL.SCAFFOLD.unit_scale,
    #         unit_scale_multiplier=self.config.MODEL.SCAFFOLD.unit_scale_multiplier,
    #         is_2dgs=self.config.MODEL.gaussian_type == "2d",
    #     )

    @torch.no_grad()
    def init_scaffold_test(self, scene_id: str, test_mode: bool = False) -> ScaffoldGSFull:
        assert test_mode, "Test mode must be True in init_scaffold_test"
        crop_points = True
        xyz, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel = self.val_dataset.load_voxelized_colmap_points(
            scene_id,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
            crop_points=crop_points,
        )
        xyz = torch.from_numpy(xyz).float().to(self.device)
        rgb = torch.from_numpy(rgb).float().to(self.device)
        xyz_voxel = torch.from_numpy(xyz_voxel).float().to(self.device)
        xyz_offset = torch.from_numpy(xyz_offset).float().to(self.device)
        voxel_to_world = torch.from_numpy(world_to_voxel).float().to(self.device).inverse()
        bbox = torch.from_numpy(bbox).float().to(self.device)
        bbox_voxel = torch.from_numpy(bbox_voxel).int().to(self.device)

        scaffolds = self.init_scaffold_train_batch(
            [xyz],
            [rgb],
            [xyz_voxel],
            [voxel_to_world],
            [bbox],
            [bbox_voxel],
            test_mode=test_mode,
        )
        return scaffolds[0]

    def _build_init_latents(
        self,
        xyz_voxel_orig: torch.Tensor,
        rgb_orig: torch.Tensor,
        xyz_voxel_init: torch.Tensor,
        scale: torch.Tensor,
        opacity: torch.Tensor,
        rgb: torch.Tensor,
        rotation: torch.Tensor,
        voxel_to_world: torch.Tensor,
    ):
        if "rgb" not in self.config.MODEL.INIT.init_type:
            # Use averaged color from the original points
            dist, indices, _ = ops.knn_points(
                xyz_voxel_init.unsqueeze(0).float(),
                xyz_voxel_orig.unsqueeze(0).float(),
                K=8,
                norm=2,
                return_nn=False,
            )
            dist = dist[0]              # (N, K)
            indices = indices[0]
            rgb = rgb_orig[indices, :]   # (N, K, 3)
            # Average the color based on the distance
            weights = 1 / (dist + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            rgb = (rgb * weights.unsqueeze(-1)).sum(dim=1)

        if "opacity" not in self.config.MODEL.INIT.init_type:
            opacity = torch.ones_like(opacity) * self.config.MODEL.GSPLAT.init_opacity
            opacity = inverse_sigmoid(opacity)

        if "scale" not in self.config.MODEL.INIT.init_type:
            scale = torch.ones_like(scale) * self.config.MODEL.SCAFFOLD.unit_scale_multiplier * self.config.MODEL.SCAFFOLD.voxel_size
            scale = torch.clamp(scale, min=1e-8, max=1e2)
            scale = scale / self.config.MODEL.SCAFFOLD.voxel_size
            scale = inverse_softplus(scale)

        if "normal" not in self.config.MODEL.INIT.init_type:
            rotation = torch.zeros_like(rotation)
            rotation[:, 0] = 1.0
        else:
            # Transform the rotation from AABB voxel space to world space
            voxel_to_world_rot = voxel_to_world[None, :3, :3] / self.config.MODEL.SCAFFOLD.voxel_size
            voxel_to_world_rot = transforms3d.matrix_to_quaternion(voxel_to_world_rot)

            # assert self.config.MODEL.INIT.init_rot_reverse, "init_rot_reverse must be True"
            # if not self.config.MODEL.INIT.init_rot_reverse:
            #     # TODO: Have to think about the order
            #     rotation = transforms3d.quaternion_multiply(rotation, voxel_to_world_rot)
            # else:
            #     # <- this is correct
            #     rotation = transforms3d.quaternion_multiply(voxel_to_world_rot, rotation)
            rotation = transforms3d.quaternion_multiply(voxel_to_world_rot, rotation)

        xyz_offsets = torch.zeros(rgb.shape[0], 3, device=rgb.device)

        latent = torch.cat([xyz_offsets, scale, rotation, opacity, rgb], dim=1)
        zero_latent = torch.zeros(
            rgb.shape[0],
            self.config.MODEL.SCAFFOLD.hidden_dim - latent.shape[1],
            device=rgb.device,
        )
        latent = torch.cat([latent, zero_latent], dim=-1)
        return latent

    @torch.no_grad()
    def _densify(
        self,
        xyz_voxel: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        voxel_to_world: torch.Tensor,
        bbox: torch.Tensor,
        bbox_voxel: Optional[torch.Tensor] = None,
        nn_color: bool = True,
        test_mode: bool = False,
    ):
        N = xyz_voxel.shape[0]
        bxyz, lengths = xyz_list_to_bxyz([xyz_voxel])
        feature = torch.ones(N, 1, device=xyz_voxel.device, dtype=torch.float32)

        sparse_tensor = ME.SparseTensor(
            coordinates=bxyz,
            features=feature,
        )
        # For full sgnn trainer, this is the timestamp that used
        init_outputs = self.initializer(sparse_tensor, gt_coords=bxyz)
        output_sparse = init_outputs["out"]
        # Compute the color using knn
        xyz_voxel_out = output_sparse.C[:, 1:].clone()

        if bbox_voxel is not None:
            # Filter the points outside the bbox
            valid_mask = (xyz_voxel_out >= bbox_voxel[0, :]).all(dim=1) & (xyz_voxel_out <= bbox_voxel[1, :]).all(dim=1)
            xyz_voxel_out = xyz_voxel_out[valid_mask]

        # Use the normal to help initialize the scaffold
        normal_pred = output_sparse.F
        normal_out = apply_rotation(normal_pred, voxel_to_world[:3, :3])
        normal_out = torch.nn.functional.normalize(normal_out, p=2, dim=-1)

        # if test_mode:
        #     save_normal_ply("normal_pred.ply", xyz_voxel_out.detach().cpu().numpy(), normal_pred.detach().cpu().numpy())
        #     breakpoint()
        # middle of the voxel
        # xyz_out = bbox[0] + (xyz_voxel_out.float() + 0.5) * self.config.MODEL.SCAFFOLD.voxel_size

        xyz_out = apply_transform(xyz_voxel_out.float(), voxel_to_world)
        # save_normal_ply("xyz_pred.ply", xyz_voxel_out.detach().cpu().numpy(), normal_pred.detach().cpu().numpy())

        if nn_color:
            dist, indices, _ = ops.knn_points(
                xyz_voxel_out.unsqueeze(0).float(),
                xyz_voxel.unsqueeze(0).float(),
                K=8,
                norm=2,
                return_nn=False,
            )
            dist = dist[0]              # (N, 10)
            indices = indices[0]
            rgb_out = rgb[indices, :]   # (N, 10, 3)
            # Average the color based on the distance
            weights = 1 / (dist + 1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True)
            rgb_out = (rgb_out * weights.unsqueeze(-1)).sum(dim=1)

            # indices = indices[0, :, 0]
            # rgb_out = rgb[indices]
            # save_ply("xyz_output.ply", xyz_out.cpu().numpy(), rgb_out.cpu().numpy())
            # breakpoint()
        else:
            # Random assign color
            rgb_out = torch.rand(xyz_voxel_out.shape[0], 3, device=xyz_voxel.device, dtype=torch.float32)
        # print(f"Before densify: {xyz_voxel.shape[0]}, After densify: {xyz_voxel_out.shape[0]}")

        # # DEBUG: Save the ply file and test it
        # print(xyz.shape, xyz_out.shape, test_mode)
        # save_ply("before.ply", xyz.cpu().numpy(), rgb.cpu().numpy())
        # save_ply("after.ply", xyz_out.cpu().numpy(), rgb_out.cpu().numpy())
        # save_ply("before_voxel.ply", xyz_voxel.cpu().numpy(), rgb.cpu().numpy())
        # save_ply("after_voxel.ply", xyz_voxel_out.cpu().numpy(), rgb_out.cpu().numpy())

        # breakpoint()

        if xyz_out.shape[0] > self.config.DATASET.max_num_points and not test_mode:
            indices = torch.randperm(xyz_out.shape[0])[:self.config.DATASET.max_num_points]
            xyz_out = xyz_out[indices]
            rgb_out = rgb_out[indices]
            xyz_voxel_out = xyz_voxel_out[indices]
            normal_out = normal_out[indices]

        if test_mode and xyz_out.shape[0] > self.config.DATASET.eval_max_num_points:
            # Prevent the out-of-memory during evaluation
            indices = torch.randperm(xyz_out.shape[0])[:self.config.DATASET.eval_max_num_points]
            xyz_out = xyz_out[indices]
            rgb_out = rgb_out[indices]
            xyz_voxel_out = xyz_voxel_out[indices]
            normal_out = normal_out[indices]
            # print(f"Downsample to {self.config.DATASET.max_num_points}")

        return xyz_voxel_out, xyz_out, rgb_out, normal_out

    def train_step(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        if self.config.MODEL.DENSIFIER.enable:
            if self.config.MODEL.DENSIFIER.train_densify_only_steps > 0 and step_idx < self.config.MODEL.DENSIFIER.train_densify_only_steps:
                return self.train_step_densifier_only(batch_cpu, point_batch, inner_step_idx, step_idx)
            else:
                return self.train_step_end_to_end(batch_cpu, point_batch, inner_step_idx, step_idx)
        else:
            return self.train_step_optimizer_only(batch_cpu, point_batch, inner_step_idx, step_idx)

    def train_step_optimizer_only(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)

        other_stuff = {}        # For debugging purposes
        loss_dict = {}
        metrics_dict = {}

        batch_gpu = self.move_to_device(batch_cpu, self.device)
        params: List[Dict[torch.Tensor]] = [s.get_raw_params() for s in self.scaffolds]
        grads: List[Dict[torch.Tensor]] = []

        for i in range(len(params)):
            grad_dict = self.get_params_grad(self.scaffolds[i], params[i], self.current_train_scene_ids[i])
            grads.append(grad_dict["grad_input"])
            # We don't want to update the scaffold parameters
            params[i] = {k: v.detach() for k, v in params[i].items()}

        bxyz, input_sizes = xyz_list_to_bxyz([x.xyz_voxel for x in self.scaffolds])
        size_acc = np.cumsum([0] + input_sizes)

        # Concate all the params and grads
        params = {k: torch.cat([p[k] for p in params], dim=0) for k in params[0].keys()}
        grads = {k: torch.cat([g[k] for g in grads], dim=0) for k in grads[0].keys()}

        outputs = self._model(bxyz, params, grads, timestamp=timestamp)

        with torch.no_grad():
            # Record the max value of the model
            metrics_dict["max_model_outputs"] = torch.max(outputs["latent"].abs())

        # Update the scaffold parameters
        for key in params.keys():
            # outputs[key] = outputs[key] * self.config.MODEL.OPT.output_scale + params[key]
            outputs[key] = (
                outputs[key]
                * self.config.MODEL.OPT.output_scale
                * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
            ) + params[key]

        with torch.no_grad():
            # Record the max value after update
            metrics_dict["max_acc_outputs"] = torch.max(outputs["latent"].abs())

        latents = []
        for i in range(len(self.scaffolds)):
            latents.append(outputs["latent"][size_acc[i]: size_acc[i + 1]])

        # Get the gs attributes for each scene
        batch_size = batch_cpu["rgb"].shape[0]
        num_views = batch_cpu["rgb"].shape[1]
        gs_params = [None for _ in range(batch_size)]
        assert batch_size == len(self.scaffolds)

        # OUTPUTs
        images_pred = []
        normals = []
        normals_from_depth = []
        distortions = []
        depth_pred = []

        for i in range(len(self.scaffolds)):
            gs_params[i] = self.scaffold_decoder(
                latent=latents[i],
                xyz_voxel=self.scaffolds[i].xyz_voxel,
                transform=self.scaffolds[i].transform,
                voxel_size=self.scaffolds[i].voxel_size,
            )
            for key in gs_params[i]:
                gs_params[i][key] = gs_params[i][key].flatten(0, 1).contiguous()

            for j in range(num_views):
                camera = Camera(
                    fov_x=float(batch_cpu["fov_x"][i][j]),
                    fov_y=float(batch_cpu["fov_y"][i][j]),
                    height=int(batch_cpu["height"][i][j]),
                    width=int(batch_cpu["width"][i][j]),
                    image=batch_gpu["rgb"][i, j],
                    world_view_transform=batch_gpu["world_view_transform"][i, j],
                    full_proj_transform=batch_gpu["full_proj_transform"][i, j],
                    camera_center=batch_gpu["camera_center"][i, j],
                )
                render_outputs = self.rasterizer(
                    gs_params[i],
                    camera,
                    active_sh_degree=0,     # Does not matter
                )
                images_pred.append(render_outputs["render"])
                if "normal" in render_outputs:
                    normals.append(render_outputs["normal"])
                if "normal_from_depth" in render_outputs:
                    normals_from_depth.append(render_outputs["normal_from_depth"])
                if "distortion" in render_outputs:
                    distortions.append(render_outputs["distortion"])
                # Use expected depth for supervision
                if "depth_expected" in render_outputs:
                    depth_pred.append(render_outputs["depth_expected"].squeeze(0))

        images_pred = torch.stack(images_pred, dim=0)
        images_gt = batch_gpu["rgb"].flatten(0, 1)

        if self.config.MODEL.gaussian_type == "2d":
            normals = torch.stack(normals, dim=0)
            normals_from_depth = torch.stack(normals_from_depth, dim=0)
            distortions = torch.stack(distortions, dim=0)
            depth_pred = torch.stack(depth_pred, dim=0)
        else:
            normals = None
            normals_from_depth = None
            distortions = None
            depth_pred = None

        if "depth" in batch_gpu:
            depth_gt = batch_gpu["depth"].flatten(0, 1)
        else:
            depth_gt = None

        loss_dict.update(
            self.compute_outer_loss(
                images_pred,
                images_gt,
                gs_params,
                distortions,
                normals,
                normals_from_depth,
                depth_pred,
                depth_gt,
            )
        )

        with torch.no_grad():
            images_pred = torch.clamp(images_pred, 0, 1)
            images_gt = torch.clamp(images_gt, 0, 1)
            metrics_dict["psnr"] = self.psnr(images_pred, images_gt)
            metrics_dict["ssim"] = self.ssim(images_pred, images_gt)
            metrics_dict["num_points"] = np.mean([x.num_points for x in self.scaffolds])
            metrics_dict["num_gs"] = np.mean([x["xyz"].shape[0] for x in gs_params])
            metrics_dict["scale"] = torch.mean(torch.cat([x["scale"] for x in gs_params], dim=0))
            metrics_dict["opacity"] = torch.mean(torch.cat([x["opacity"] for x in gs_params], dim=0))

        if self.config.MODEL.use_identity_loss:
            identity_loss_dict, identity_metrics_dict = self.compute_identity_loss(
                latents,
                batch_cpu,
                batch_gpu,
            )
            loss_dict.update(identity_loss_dict)
            metrics_dict.update(identity_metrics_dict)

        for i in range(len(self.scaffolds)):
            self.scaffolds[i].set_raw_params({"latent": latents[i]})

        return (
            loss_dict,
            metrics_dict,
            other_stuff,
        )

    def train_step_densifier_only(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        # TODO: Update this part
        raise NotImplementedError("train_step_densifier_only is not implemented yet")

        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)

        other_stuff = {}        # For debugging purposes
        loss_dict = {}
        metrics_dict = {}

        params: List[Dict[torch.Tensor]] = [s.get_raw_params() for s in self.scaffolds]
        grads: List[Dict[torch.Tensor]] = []
        # grad_2d = []
        # grad_2d_norm = []

        for i in range(len(params)):
            # We don't update the scaffold parameters
            # Use all_zeros for the params and grads
            params[i] = {k: torch.zeros_like(v) for k, v in params[i].items()}
            grads.append({k: torch.zeros_like(v) for k, v in params[i].items()})

        # grad_2d = torch.cat(grad_2d, dim=0)

        bxyz, input_sizes = xyz_list_to_bxyz([x.xyz_voxel for x in self.scaffolds])
        input_sizes = np.cumsum([0] + input_sizes)
        bxyz_gt, gt_sizes = xyz_list_to_bxyz(point_batch["xyz_voxel_gt"])

        if self.config.MODEL.INIT.dilate_gt > 0:
            with torch.no_grad():
                bxyz_gt = dilation_sparse(bxyz_gt, self.config.MODEL.INIT.dilate_gt)

        gt_sizes = np.cumsum([0] + gt_sizes)

        # Concate all the params and grads
        params_cat = {k: torch.cat([p[k] for p in params], dim=0) for k in params[0].keys()}
        grads_cat = {k: torch.cat([g[k] for g in grads], dim=0) for k in grads[0].keys()}

        if self.config.MODEL.DENSIFIER.sample_num_per_step > 0:
            # *= batch size
            sample_n_last = self.config.MODEL.DENSIFIER.sample_num_per_step * len(self.scaffolds)
            sample_n_last //= (2 ** inner_step_idx)
            temperature = self.config.MODEL.DENSIFIER.sample_temperature
            decay = self.config.MODEL.DENSIFIER.sample_temperature_decay
            temperature = max(temperature * (decay ** step_idx), self.config.MODEL.DENSIFIER.sample_temperature_min)
        else:
            sample_n_last = None
            temperature = 1.0   # Does not matter
        if self.config.debug:
            print(f"Original points: {bxyz.shape}")

        # print(f"Original points: {bxyz.shape} inner_step={inner_step_idx}")
        dense_outputs = self.densifier(
            params_cat,
            grads_cat,
            bxyz,
            bxyz_gt,
            timestamp=timestamp,
            sample_n_last=sample_n_last,
            sample_temperature=temperature,
            # max_gt_samples_training=self.config.DATASET.max_num_points * len(self.scaffolds),
            max_gt_samples_training=sample_n_last,
            max_samples_second_last=self.config.MODEL.DENSIFIER.sample_num_per_step * len(self.scaffolds) // 2,
        )

        output_sparse = dense_outputs["out"]
        out_cls = dense_outputs["out_cls"]
        targets = dense_outputs["targets"]
        ignores = dense_outputs["ignores"]
        ignore_final = dense_outputs["ignore_final"]

        # Select only those that exists
        before_prune = output_sparse
        output_sparse = self.prune(output_sparse, ~ignore_final)
        metrics_dict["new_points"] = output_sparse.shape[0]

        if self.config.debug:
            print(f"Generating new points: {output_sparse.shape[0]} / {before_prune.shape[0]}")
        # print(f"Generating new points: {output_sparse.shape} inner_step={inner_step_idx}")
        xyz_list_new, latent_list_new = output_sparse.decomposed_coordinates_and_features
        grad_list_new = [torch.zeros_like(x) for x in latent_list_new]

        # Concatenate the new points to the existing scaffold
        for i in range(len(self.scaffolds)):
            xyz_list_new[i] = torch.cat([self.scaffolds[i].xyz_voxel, xyz_list_new[i]], dim=0)
            latent_list_new[i] = torch.cat([params[i]["latent"], latent_list_new[i]], dim=0)
            grad_list_new[i] = torch.cat([grads[i]["latent"], grad_list_new[i]], dim=0)

        bxyz, input_sizes = xyz_list_to_bxyz(xyz_list_new)
        input_sizes = np.cumsum([0] + input_sizes)
        params_new = [{"latent": x} for x in latent_list_new]
        grads_new = [{"latent": x} for x in grad_list_new]

        params_cat = {k: torch.cat([p[k] for p in params_new], dim=0) for k in params_new[0].keys()}
        grads_cat = {k: torch.cat([g[k] for g in grads_new], dim=0) for k in grads_new[0].keys()}
        outputs = self._model(bxyz, params_cat, grads_cat, timestamp=timestamp)

        # print(f"After points: {bxyz.shape} inner_step={inner_step_idx}")

        # Compute occ loss
        # occ_loss = 0.0
        # assert self.config.MODEL.OPT.occ_mult > 0
        # for idx, (pred, gt, ignore_mask) in enumerate(zip(out_cls, targets, ignores)):
        #     pred = pred.F.flatten(0, 1)
        #     gt = gt.float()

        #     if self.config.MODEL.DENSIFIER.loss_ignore_exist:
        #         # Ignore the existing points in "bxyz" in each level
        #         # Could be a problem where most of the voxels are ignored
        #         valid_mask = ~ignore_mask
        #         if self.config.debug:
        #             ratio = valid_mask.sum() / valid_mask.numel()
        #             print(f"{idx}: valid: {valid_mask.sum()}/{valid_mask.numel()}, ratio: {ratio:.2f}")
        #         pred = pred[valid_mask]
        #         gt = gt[valid_mask]

        #     if self.config.MODEL.INIT.use_balance_weight:
        #         num_pos = gt.sum()
        #         num_neg = gt.numel() - num_pos
        #         pos_weight = num_neg / num_pos
        #         pos_weight = pos_weight.detach()
        #         if self.config.debug:
        #             print(f"{idx}: shape: {gt.shape}, weight: {pos_weight:.2f}")
        #     else:
        #         pos_weight = None
        #     occ = nn.functional.binary_cross_entropy_with_logits(
        #         pred,
        #         gt,
        #         pos_weight=pos_weight,
        #         # weight=loss_mask,
        #     )

        #     occ_loss += occ / len(targets)
        occ_loss = self.compute_occ_loss(out_cls, targets, ignores)
        loss_dict["occ_loss"] = occ_loss * self.config.MODEL.OPT.occ_mult
        if self.config.debug:
            if step_idx % 5 == 0:
                print(f"Occ loss: {occ_loss.item():.4f}")

        # Update the scaffold parameters
        latent_delta = (
            outputs["latent"]
            * self.config.MODEL.OPT.output_scale
            * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
        )

        params_cat["latent"] = params_cat["latent"] + latent_delta

        # for key in params.keys():
        #     # outputs[key] = outputs[key] * self.config.MODEL.OPT.output_scale + params[key]
        #     outputs[key] = (
        #         outputs[key]
        #         * self.config.MODEL.OPT.output_scale
        #         * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
        #     ) + params[key]

        with torch.no_grad():
            metrics_dict["max_model_outputs"] = torch.max(outputs["latent"])
            metrics_dict["max_acc_outputs"] = torch.max(params_cat["latent"])

        latent_per_scene = []
        for i in range(len(self.scaffolds)):
            latent_per_scene.append(params_cat["latent"][input_sizes[i]:input_sizes[i+1]])

        with torch.no_grad():
            xyz_voxel_gt = point_batch["xyz_voxel_gt"]
            chamfer_init = chamfer_dist(self.scaffolds[0].xyz_voxel, xyz_voxel_gt[0], norm=1)
            metrics_dict["chamfer_init"] = chamfer_init.item()
            chamfer_full = chamfer_dist(xyz_list_new[0], xyz_voxel_gt[0], norm=1)
            metrics_dict["chamfer_full"] = chamfer_full.item()

            # if self.config.debug and step_idx % 5 == 0:
            #     xyz_voxel_gt = xyz_voxel_gt[0].detach().cpu().numpy()
            #     xyz_voxel_init = self.scaffolds[0].xyz_voxel.detach().cpu().numpy()
            #     xyz_voxel_pred = xyz_list_new[0].detach().cpu().numpy()

            #     xyz = point_batch["xyz"][0].detach().cpu().numpy()
            #     xyz_gt = point_batch["xyz_gt"][0].detach().cpu().numpy()
            #     save_ply(self.output_dir / "val" / "colmap_xyz.ply", xyz, point_batch["rgb"][0].detach().cpu().numpy())
            #     save_ply(self.output_dir / "val" / "gt_xyz.ply", xyz_gt, np.ones_like(xyz_gt))

            #     save_ply(self.output_dir / "val" / "gt_voxel.ply", xyz_voxel_gt, np.ones_like(xyz_voxel_gt))
            #     save_ply(self.output_dir / "val" / "init_voxel.ply", xyz_voxel_init, np.ones_like(xyz_voxel_init))
            #     save_ply(self.output_dir / "val" / "pred_voxel.ply", xyz_voxel_pred, np.ones_like(xyz_voxel_pred))

            #     params_tmp = self.identity_decoder(
            #         latent=self.scaffolds[0].get_raw_params()["latent"],
            #         xyz_voxel=self.scaffolds[0].xyz_voxel,
            #         # latent=self.scaffolds[i].get_raw_params()["latent"],
            #         # xyz_voxel=self.scaffolds[i].xyz_voxel,
            #         transform=self.scaffolds[0].transform,
            #         voxel_size=self.scaffolds[0].voxel_size,
            #     )
            #     xyz = params_tmp["xyz"].squeeze(1).detach().cpu().numpy()
            #     rgb = params_tmp["rgb"].squeeze(1).detach().cpu().numpy()
            #     print(xyz.shape, rgb.shape)
            #     save_ply(self.output_dir / "val" / "init_xyz.ply", xyz, rgb)

            #     params_tmp = self.identity_decoder(
            #         latent=latent_per_scene[0],
            #         xyz_voxel=xyz_list_new[0],
            #         transform=self.scaffolds[0].transform,
            #         voxel_size=self.scaffolds[0].voxel_size,
            #     )
            #     xyz = params_tmp["xyz"].squeeze(1).detach().cpu().numpy()
            #     rgb = params_tmp["rgb"].squeeze(1).detach().cpu().numpy()
            #     save_ply(self.output_dir / "val" / "pred_xyz.ply", xyz, rgb)

        # Don't compute any gaussian related loss
        if self.config.debug:
            print(f"Only training densifier: step={step_idx}, inner_step={inner_step_idx}")

        for i in range(len(self.scaffolds)):
            self.scaffolds[i].set_raw_params_and_xyz({"latent": latent_per_scene[i]}, xyz_list_new[i])

        return loss_dict, metrics_dict, other_stuff

    def train_step_end_to_end(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        assert self.config.MODEL.DENSIFIER.enable
        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)

        other_stuff = {}        # For debugging purposes
        loss_dict = {}
        metrics_dict = {}

        batch_gpu = self.move_to_device(batch_cpu)
        params: List[Dict[torch.Tensor]] = [s.get_raw_params() for s in self.scaffolds]
        grads: List[Dict[torch.Tensor]] = []
        grad_2d = []
        grad_2d_norm = []

        for i in range(len(params)):
            grad_dict = self.get_params_grad(self.scaffolds[i], params[i], self.current_train_scene_ids[i])
            grads.append(grad_dict["grad_input"])
            grad_2d.append(grad_dict["grad_2d"])
            grad_2d_norm.append(grad_dict["grad_2d_norm"])
            # We don't want to update the scaffold parameters
            params[i] = {k: v.detach() for k, v in params[i].items()}
        grad_2d = torch.cat(grad_2d, dim=0)

        bxyz, input_sizes = xyz_list_to_bxyz([x.xyz_voxel for x in self.scaffolds])
        input_sizes = np.cumsum([0] + input_sizes)
        bxyz_gt, gt_sizes = xyz_list_to_bxyz(point_batch["xyz_voxel_gt"])
        gt_sizes = np.cumsum([0] + gt_sizes)

        if self.config.MODEL.INIT.dilate_gt > 0:
            raise NotImplementedError("TODO: ERROR. the gt_sizes might need to be changed")
            with torch.no_grad():
                bxyz_gt = dilation_sparse(bxyz_gt, self.config.MODEL.INIT.dilate_gt)

        # Concate all the params and grads
        params_cat = {k: torch.cat([p[k] for p in params], dim=0) for k in params[0].keys()}
        grads_cat = {k: torch.cat([g[k] for g in grads], dim=0) for k in grads[0].keys()}

        if self.config.MODEL.DENSIFIER.sample_num_per_step > 0:
            # *= batch size
            sample_n_last = self.config.MODEL.DENSIFIER.sample_num_per_step * len(self.scaffolds)
            sample_n_last //= (2 ** inner_step_idx)
            temperature = self.config.MODEL.DENSIFIER.sample_temperature
            decay = self.config.MODEL.DENSIFIER.sample_temperature_decay
            temperature = max(temperature * (decay ** step_idx), self.config.MODEL.DENSIFIER.sample_temperature_min)
        else:
            sample_n_last = None
            temperature = 1.0   # Does not matter
        if self.config.debug:
            print(f"Original points: {bxyz.shape}")

        # print(f"Original points: {bxyz.shape} inner_step={inner_step_idx}")
        dense_outputs = self.densifier(
            params_cat,
            grads_cat,
            bxyz,
            bxyz_gt,
            timestamp=timestamp,
            sample_n_last=sample_n_last,
            sample_temperature=temperature,
            max_gt_samples_training=sample_n_last,
            max_samples_second_last=self.config.MODEL.DENSIFIER.sample_num_per_step * len(self.scaffolds) // 2,
        )

        output_sparse = dense_outputs["out"]
        out_cls = dense_outputs["out_cls"]
        targets = dense_outputs["targets"]
        ignores = dense_outputs["ignores"]
        ignore_final = dense_outputs["ignore_final"]

        # Select only those that exists
        before_prune = output_sparse
        output_sparse = self.prune(output_sparse, ~ignore_final)
        metrics_dict["new_points"] = output_sparse.shape[0]

        if self.config.debug:
            print(f"Generating new points: {output_sparse.shape[0]} / {before_prune.shape[0]}")
        # print(f"Generating new points: {output_sparse.shape} inner_step={inner_step_idx}")
        xyz_list_new, latent_list_new = output_sparse.decomposed_coordinates_and_features
        grad_list_new = [torch.zeros_like(x) for x in latent_list_new]

        # Concatenate the new points to the existing scaffold
        for i in range(len(self.scaffolds)):
            xyz_list_new[i] = torch.cat([self.scaffolds[i].xyz_voxel, xyz_list_new[i]], dim=0)
            latent_list_new[i] = torch.cat([params[i]["latent"], latent_list_new[i]], dim=0)
            grad_list_new[i] = torch.cat([grads[i]["latent"], grad_list_new[i]], dim=0)

        bxyz, input_sizes = xyz_list_to_bxyz(xyz_list_new)
        input_sizes = np.cumsum([0] + input_sizes)
        params_new = [{"latent": x} for x in latent_list_new]
        grads_new = [{"latent": x} for x in grad_list_new]

        params_cat = {k: torch.cat([p[k] for p in params_new], dim=0) for k in params_new[0].keys()}
        grads_cat = {k: torch.cat([g[k] for g in grads_new], dim=0) for k in grads_new[0].keys()}

        if self.config.MODEL.DENSIFIER.recompute_grad:
            for i in range(len(self.scaffolds)):
                self.scaffolds[i].set_raw_params_and_xyz(params_new[i], xyz_list_new[i])
                grad_dict = self.get_params_grad(self.scaffolds[i], params_new[i], self.current_train_scene_ids[i])
                grad_list_new[i] = grad_dict["grad_input"]

        outputs = self._model(
            bxyz,
            params_cat,
            grads_cat,
            timestamp=timestamp,
            remap_mapping=None,
        )

        # print(f"After points: {bxyz.shape} inner_step={inner_step_idx}")

        # Compute occ loss
        # occ_loss = 0.0
        # assert self.config.MODEL.OPT.occ_mult > 0
        # for idx, (pred, gt, ignore_mask) in enumerate(zip(out_cls, targets, ignores)):
        #     pred = pred.F.flatten(0, 1)
        #     gt = gt.float()

        #     if self.config.MODEL.DENSIFIER.loss_ignore_exist:
        #         # Ignore the existing points in "bxyz" in each level
        #         # Could be a problem where most of the voxels are ignored
        #         valid_mask = ~ignore_mask
        #         if self.config.debug:
        #             ratio = valid_mask.sum() / valid_mask.numel()
        #             print(f"{idx}: valid: {valid_mask.sum()}/{valid_mask.numel()}, ratio: {ratio:.2f}")
        #         pred = pred[valid_mask]
        #         gt = gt[valid_mask]

        #     if self.config.MODEL.INIT.use_balance_weight:
        #         num_pos = gt.sum()
        #         num_neg = gt.numel() - num_pos
        #         pos_weight = num_neg / num_pos
        #         pos_weight = pos_weight.detach()
        #         if self.config.debug:
        #             print(f"{idx}: shape: {gt.shape}, weight: {pos_weight:.2f}")
        #     else:
        #         pos_weight = None
        #     occ = nn.functional.binary_cross_entropy_with_logits(
        #         pred,
        #         gt,
        #         pos_weight=pos_weight,
        #         # weight=loss_mask,
        #     )

        #     occ_loss += occ / len(targets)

        occ_loss = self.compute_occ_loss(out_cls, targets, ignores)
        loss_dict["occ_loss"] = occ_loss * self.config.MODEL.OPT.occ_mult
        if self.config.debug:
            if step_idx % 5 == 0:
                print(f"Occ loss: {occ_loss.item():.4f}")

        # Update the scaffold parameters
        latent_delta = (
            outputs["latent"]
            * self.config.MODEL.OPT.output_scale
            * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
        )

        params_cat["latent"] = params_cat["latent"] + latent_delta

        # for key in params.keys():
        #     # outputs[key] = outputs[key] * self.config.MODEL.OPT.output_scale + params[key]
        #     outputs[key] = (
        #         outputs[key]
        #         * self.config.MODEL.OPT.output_scale
        #         * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
        #     ) + params[key]

        with torch.no_grad():
            metrics_dict["max_model_outputs"] = torch.max(outputs["latent"])
            metrics_dict["max_acc_outputs"] = torch.max(params_cat["latent"])

        latent_per_scene = []
        for i in range(len(self.scaffolds)):
            latent_per_scene.append(params_cat["latent"][input_sizes[i]: input_sizes[i + 1]])

        with torch.no_grad():
            xyz_voxel_gt = point_batch["xyz_voxel_gt"]
            chamfer_init = chamfer_dist(self.scaffolds[0].xyz_voxel, xyz_voxel_gt[0], norm=1)
            metrics_dict["chamfer_init"] = chamfer_init.item()
            chamfer_full = chamfer_dist(xyz_list_new[0], xyz_voxel_gt[0], norm=1)
            metrics_dict["chamfer_full"] = chamfer_full.item()

            # if self.config.debug and step_idx % 5 == 0:
            #     xyz_voxel_gt = xyz_voxel_gt[0].detach().cpu().numpy()
            #     xyz_voxel_init = self.scaffolds[0].xyz_voxel.detach().cpu().numpy()
            #     xyz_voxel_pred = xyz_list_new[0].detach().cpu().numpy()

            #     xyz = point_batch["xyz"][0].detach().cpu().numpy()
            #     xyz_gt = point_batch["xyz_gt"][0].detach().cpu().numpy()
            #     save_ply(self.output_dir / "val" / "colmap_xyz.ply", xyz, point_batch["rgb"][0].detach().cpu().numpy())
            #     save_ply(self.output_dir / "val" / "gt_xyz.ply", xyz_gt, np.ones_like(xyz_gt))

            #     save_ply(self.output_dir / "val" / "gt_voxel.ply", xyz_voxel_gt, np.ones_like(xyz_voxel_gt))
            #     save_ply(self.output_dir / "val" / "init_voxel.ply", xyz_voxel_init, np.ones_like(xyz_voxel_init))
            #     save_ply(self.output_dir / "val" / "pred_voxel.ply", xyz_voxel_pred, np.ones_like(xyz_voxel_pred))

            #     params_tmp = self.identity_decoder(
            #         latent=self.scaffolds[0].get_raw_params()["latent"],
            #         xyz_voxel=self.scaffolds[0].xyz_voxel,
            #         # latent=self.scaffolds[i].get_raw_params()["latent"],
            #         # xyz_voxel=self.scaffolds[i].xyz_voxel,
            #         transform=self.scaffolds[0].transform,
            #         voxel_size=self.scaffolds[0].voxel_size,
            #     )
            #     xyz = params_tmp["xyz"].squeeze(1).detach().cpu().numpy()
            #     rgb = params_tmp["rgb"].squeeze(1).detach().cpu().numpy()
            #     print(xyz.shape, rgb.shape)
            #     save_ply(self.output_dir / "val" / "init_xyz.ply", xyz, rgb)

            #     params_tmp = self.identity_decoder(
            #         latent=latent_per_scene[0],
            #         xyz_voxel=xyz_list_new[0],
            #         transform=self.scaffolds[0].transform,
            #         voxel_size=self.scaffolds[0].voxel_size,
            #     )
            #     xyz = params_tmp["xyz"].squeeze(1).detach().cpu().numpy()
            #     rgb = params_tmp["rgb"].squeeze(1).detach().cpu().numpy()
            #     save_ply(self.output_dir / "val" / "pred_xyz.ply", xyz, rgb)

        # Get the gs attributes for each scene
        batch_size = batch_cpu["rgb"].shape[0]
        num_views = batch_cpu["rgb"].shape[1]
        gs_params = [None for _ in range(batch_size)]
        assert batch_size == len(self.scaffolds)

        # OUTPUTs
        images_pred = []
        normals = []
        normals_from_depth = []
        distortions = []
        depth_pred = []

        for i in range(len(self.scaffolds)):
            if self.config.debug:
                print(f"Decoding {i}-th latent with size: {latent_per_scene[i].shape}")

            gs_params[i] = self.scaffold_decoder(
                latent=latent_per_scene[i],
                xyz_voxel=xyz_list_new[i],
                # latent=self.scaffolds[i].get_raw_params()["latent"],
                # xyz_voxel=self.scaffolds[i].xyz_voxel,
                transform=self.scaffolds[i].transform,
                voxel_size=self.scaffolds[i].voxel_size,
            )
            for key in gs_params[i]:
                gs_params[i][key] = gs_params[i][key].flatten(0, 1).contiguous()
                # if self.config.debug:
                #     print(f"{key}: {gs_params[i][key].shape}, min: {gs_params[i][key].min()}, max: {gs_params[i][key].max()}")

            for j in range(num_views):
                camera = Camera(
                    fov_x=float(batch_cpu["fov_x"][i][j]),
                    fov_y=float(batch_cpu["fov_y"][i][j]),
                    height=int(batch_cpu["height"][i][j]),
                    width=int(batch_cpu["width"][i][j]),
                    image=batch_gpu["rgb"][i, j],
                    world_view_transform=batch_gpu["world_view_transform"][i, j],
                    full_proj_transform=batch_gpu["full_proj_transform"][i, j],
                    camera_center=batch_gpu["camera_center"][i, j],
                )
                render_outputs = self.rasterizer(
                    gs_params[i],
                    camera,
                    active_sh_degree=0,     # Does not matter
                )
                images_pred.append(render_outputs["render"])
                if "normal" in render_outputs:
                    normals.append(render_outputs["normal"])
                if "normal_from_depth" in render_outputs:
                    normals_from_depth.append(render_outputs["normal_from_depth"])
                if "distortion" in render_outputs:
                    distortions.append(render_outputs["distortion"])
                # Use expected depth for supervision
                if "depth_expected" in render_outputs:
                    depth_pred.append(render_outputs["depth_expected"].squeeze(0))

        images_pred = torch.stack(images_pred, dim=0)
        images_gt = batch_gpu["rgb"].flatten(0, 1)

        if self.config.MODEL.gaussian_type == "2d":
            normals = torch.stack(normals, dim=0)
            normals_from_depth = torch.stack(normals_from_depth, dim=0)
            distortions = torch.stack(distortions, dim=0)
            depth_pred = torch.stack(depth_pred, dim=0)
        else:
            normals = None
            normals_from_depth = None
            distortions = None
            depth_pred = None

        if "depth" in batch_gpu:
            depth_gt = batch_gpu["depth"].flatten(0, 1)
        else:
            depth_gt = None

        loss_dict.update(
            self.compute_outer_loss(
                images_pred,
                images_gt,
                gs_params,
                distortions,
                normals,
                normals_from_depth,
                depth_pred,
                depth_gt,
            )
        )

        with torch.no_grad():
            images_pred = torch.clamp(images_pred, 0, 1)
            images_gt = torch.clamp(images_gt, 0, 1)
            metrics_dict["psnr"] = self.psnr(images_pred, images_gt)
            metrics_dict["ssim"] = self.ssim(images_pred, images_gt)
            metrics_dict["num_points"] = np.mean([x.num_points for x in self.scaffolds])
            metrics_dict["num_gs"] = np.mean([x["xyz"].shape[0] for x in gs_params])

        if self.config.MODEL.use_identity_loss:
            identity_loss_dict, identity_metrics_dict = self.compute_identity_loss(
                latent_per_scene,
                batch_cpu,
                batch_gpu,
                xyz_voxels=xyz_list_new,
            )
            loss_dict.update(identity_loss_dict)
            metrics_dict.update(identity_metrics_dict)

        for i in range(len(self.scaffolds)):
            self.scaffolds[i].set_raw_params_and_xyz({"latent": latent_per_scene[i]}, xyz_list_new[i])

        return (
            loss_dict,
            metrics_dict,
            other_stuff,
        )

    def validate_without_densifier(self, step_idx: int, save_path: Optional[str] = None) -> Dict[str, float]:
        self.model.eval()
        self.scaffold_decoder.eval()
        # TODO: Implement validation without densifier

        self.model.train()
        self.scaffold_decoder.train()

    def validate(self, step_idx: int, save_path: Optional[str] = None) -> Dict[str, float]:
        if not self.config.MODEL.DENSIFIER.enable:
            return self.validate_without_densifier(step_idx, save_path)

        self.model.eval()
        self.densifier.eval()
        self.scaffold_decoder.eval()

        metrics_list_init = []
        metrics_list_final = []

        for scene_idx, scene_id in enumerate(tqdm(self.val_dataset.scene_list, desc="Validation")):
            scaffold = self.init_scaffold_test(scene_id, test_mode=True)
            xyz_voxel_init = scaffold.xyz_voxel.clone()

            init_metrics = self.evaluate_gs(
                scaffold=scaffold,
                scene_id=scene_id,
                save_path=save_path,
                save_file_suffix="init",
                use_identity=True,
            )
            metrics_list_init.extend(init_metrics)

            xyz_gt, rgb_gt, xyz_voxel_gt, *_ = self.val_dataset.load_voxelized_mesh_points(
                scene_id,
                voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
                bbox_min=scaffold.bbox[0].cpu().numpy(),
                bbox_max=scaffold.bbox[1].cpu().numpy(),
                load_normal=False,
            )
            xyz_voxel_gt = torch.from_numpy(xyz_voxel_gt).int().to(self.device)
            # bxyz_gt, _ = xyz_list_to_bxyz([xyz_voxel_gt])

            chamfer_init = chamfer_dist(scaffold.xyz_voxel, xyz_voxel_gt, norm=1).item()
            chamfer_gt_to_pred_init = chamfer_dist(xyz_voxel_gt, scaffold.xyz_voxel, norm=1, single_directional=True).item()

            # Run the quicksplat optimization
            all_new_xyz_voxels = []
            num_new_points = 0
            for inner_step_idx in range(self.num_inner_steps):
                timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)
                params = scaffold.get_raw_params()

                grad_dict = self.get_params_grad(scaffold, params, scene_id, fixed=True)
                grads = grad_dict["grad_input"]
                grad_2d = grad_dict["grad_2d"]

                # We don't update the scaffold parameters
                params = {k: v.detach() for k, v in params.items()}

                if self.config.MODEL.DENSIFIER.train_densify_only_steps > 0 and step_idx < self.config.MODEL.DENSIFIER.train_densify_only_steps:
                    # Don't use the params and grads when training densifier only
                    params = {k: torch.zeros_like(v) for k, v in params.items()}
                    grads = {k: torch.zeros_like(v) for k, v in grads.items()}

                bxyz, _ = xyz_list_to_bxyz([scaffold.xyz_voxel])

                if self.config.MODEL.DENSIFIER.sample_num_per_step_eval > 0:
                    sample_n_last = self.config.MODEL.DENSIFIER.sample_num_per_step_eval
                    sample_n_last //= (2 ** inner_step_idx)
                    samples_second_last = self.config.MODEL.DENSIFIER.sample_num_per_step_eval // 2
                elif self.config.MODEL.DENSIFIER.sample_num_per_step > 0:
                    sample_n_last = self.config.MODEL.DENSIFIER.sample_num_per_step
                    sample_n_last //= (2 ** inner_step_idx)
                    samples_second_last = self.config.MODEL.DENSIFIER.sample_num_per_step // 2
                else:
                    sample_n_last = None
                    samples_second_last = None

                # print(f"Original points: {bxyz.shape} inner_step={inner_step_idx}")
                # TODO: Add more points than training for evaluation
                with torch.no_grad():
                    dense_outputs = self.densifier(
                        params,
                        grads,
                        bxyz,
                        bxyz,
                        timestamp=timestamp,
                        sample_n_last=sample_n_last,
                        # temperature is not used in eval if eval_randomness = False
                        # else use sample_temperature_min
                        sample_temperature=self.config.MODEL.DENSIFIER.sample_temperature_min,
                        # max_gt_samples_training=None,       # gt is not used during eval
                        # TODO: Try this?
                        max_gt_samples_training=sample_n_last,
                        # max_gt_samples_training=self.config.DATASET.eval_max_num_points,
                        # TODO: Might need to use this to save memory
                        max_samples_second_last=samples_second_last,
                    )

                    output_sparse = dense_outputs["out"]
                    ignore_final = dense_outputs["ignore_final"]
                    # Select only those that exists
                    before_prune = output_sparse
                    output_sparse = self.prune(output_sparse, ~ignore_final)
                    num_new_points += output_sparse.shape[0]
                    if self.config.debug:
                        print(f"Generating new points: {output_sparse.shape[0]} / {before_prune.shape[0]}")

                    xyz_new, latent_new = output_sparse.decomposed_coordinates_and_features
                    xyz_new = xyz_new[0]
                    all_new_xyz_voxels.append(xyz_new)
                    latent_new = latent_new[0]
                    grad_new = torch.zeros_like(latent_new)

                    xyz_new = torch.cat([scaffold.xyz_voxel, xyz_new], dim=0)
                    latent_new = torch.cat([params["latent"], latent_new], dim=0)
                    grad_new = torch.cat([grads["latent"], grad_new], dim=0)

                    if self.config.MODEL.DENSIFIER.recompute_grad:
                        scaffold.set_raw_params_and_xyz({"latent": latent_new}, xyz_new)
                        grad_dict = self.get_params_grad(scaffold, {"latent": latent_new}, scene_id, fixed=True)
                        grad_new = grad_dict["grad_input"]
                    else:
                        # The gradients for the new Gaussians are zero
                        pass

                    bxyz, _ = xyz_list_to_bxyz([xyz_new])

                    outputs = self._model(
                        bxyz,
                        {"latent": latent_new},
                        {"latent": grad_new},
                        timestamp=timestamp,
                        remap_mapping=None,
                    )
                    # print(f"After points: {bxyz.shape} inner_step={inner_step_idx}")

                    latent_delta = (
                        outputs["latent"]
                        * self.config.MODEL.OPT.output_scale
                        * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
                    )

                    latent_new = latent_new + latent_delta
                    scaffold.set_raw_params_and_xyz({"latent": latent_new}, xyz_new)
                # print(f"after inner={inner_step_idx} outputs: {outputs['latent'].shape}, params: {scaffold.get_raw_params()['latent'].shape}")

            chamfer_full = chamfer_dist(scaffold.xyz_voxel, xyz_voxel_gt, norm=1).item()
            chamfer_pred_to_gt = chamfer_dist(scaffold.xyz_voxel, xyz_voxel_gt, norm=1, single_directional=True).item()
            chamfer_gt_to_pred = chamfer_dist(xyz_voxel_gt, scaffold.xyz_voxel, norm=1, single_directional=True).item()
            # if scene_idx < 10 and save_path is not None:
            #     save_path = Path(save_path)
            #     save_path.mkdir(parents=True, exist_ok=True)

            #     if scene_idx == 0:
            #         print(f"Saving to {save_path}")

            #     xyz_voxel_init = xyz_voxel_init.cpu().numpy()
            #     save_ply(save_path / f"{scene_id}_init.ply", xyz_voxel_init, np.zeros_like(xyz_voxel_init))
            #     xyz_voxel_gt = xyz_voxel_gt.cpu().numpy()
            #     save_ply(save_path / f"{scene_id}_gt.ply", xyz_voxel_gt, rgb_gt)
            #     xyz_voxel_pred = scaffold.xyz_voxel.cpu().numpy()
            #     save_ply(save_path / f"{scene_id}_pred.ply", xyz_voxel_pred, np.zeros_like(xyz_voxel_pred))

            #     all_new_xyz_voxels = torch.cat(all_new_xyz_voxels, dim=0).cpu().numpy()
            #     save_ply(save_path / f"{scene_id}_all_new.ply", all_new_xyz_voxels, np.zeros_like(all_new_xyz_voxels))

            final_metrics = self.evaluate_gs(
                scaffold=scaffold,
                scene_id=scene_id,
                save_path=save_path,
                save_file_suffix="final",
                use_identity=False,
            )
            for x in final_metrics:
                x["chamfer_init"] = chamfer_init
                x["chamfer_gt_to_pred_init"] = chamfer_gt_to_pred_init
                x["chamfer_full"] = chamfer_full
                x["chamfer_pred_to_gt"] = chamfer_pred_to_gt
                x["chamfer_gt_to_pred"] = chamfer_gt_to_pred
                x["num_new_points"] = num_new_points

            metrics_list_final.extend(final_metrics)
            torch.cuda.empty_cache()

        final_metrics = {}
        for k, v in metrics_list_final[0].items():
            final_metrics[k] = np.mean([m[k] for m in metrics_list_final])

        for k in metrics_list_init[0].keys():
            final_metrics[f"{k}_init"] = np.mean([m[k] for m in metrics_list_init])
        self.model.train()
        self.densifier.train()
        self.scaffold_decoder.train()
        print(json.dumps(final_metrics, indent=4))
        return final_metrics
