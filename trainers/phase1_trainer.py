import json
from typing import Any, Union, Optional, List, Tuple, Dict, Literal
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm
import MinkowskiEngine as ME
import numpy as np
from pytorch3d import ops

from trainers.quicksplat_base_trainer import QuickSplatTrainer
from models.optimizer import UNetOptimizer
from models.initializer import Initializer18
from models.unet_base import ResUNetConfig
from models.scaffold_gs import ScaffoldGSFull, inverse_sigmoid, inverse_softplus
from modules.rasterizer_3d import ScaffoldRasterizer, Camera
from modules.rasterizer_2d import Scaffold2DGSRasterizer

from utils.pose import quaternion_to_normal
from utils.optimizer import Optimizer
from utils.sparse import xyz_list_to_bxyz, chamfer_dist
from utils.utils import load_state_dict_custom
from utils.rich_utils import CONSOLE


class Phase1Trainer(QuickSplatTrainer):
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

    def get_model(self) -> nn.Module:
        assert self.config.MODEL.decoder_type == "identity", "Phase1Trainer only supports identity decoder"
        self.setup_decoders()
        if self.config.MODEL.gaussian_type == "3d":
            out_channels = 3 + 3 + 4    # scale (3) + rgb (3) + rotation (4)
        else:
            out_channels = 2 + 3 + 4    # scale (2) + rgb (3) + rotation (4)
        model = Initializer18(
            in_channels=3,
            out_channels=out_channels,
            config=ResUNetConfig(),
            use_time_emb=False,
            dense_bottleneck=True,
            num_dense_blocks=self.config.MODEL.INIT.num_dense_blocks,
        )

        # Load the SGNN checkpoint as the pretrained initializer
        CONSOLE.print(f"Loading pretrained SGNN checkpoint from {self.config.TRAIN.ckpt_path}")
        ckpt_path = Path(self.config.TRAIN.ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist"
        # If the checkpoint is a directory, load the latest checkpoint
        if ckpt_path.is_dir():
            ckpt_path = sorted(ckpt_path.glob("*.ckpt"))[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        load_state_dict_custom(model, ckpt["model"])
        return model

    def setup_optimizers(self):
        assert self.config.MODEL.decoder_type == "identity", "Phase1Trainer only supports identity decoder"
        # No decoder parameters when using identity decoder
        param_dict = {
            "model": self.model.parameters(),
        }
        self.optimizer = Optimizer(param_dict, self.config.OPTIMIZER)

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

        if self.config.MODEL.input_type == "colmap+completion":
            xyz_voxel_init = []
            rgb_init = [None for _ in range(batch_size)]
        else:
            xyz_voxel_init = xyz_voxel_list
            rgb_init = rgb_list

        for i in range(batch_size):
            # For loop version: slower but use less memory
            if self.config.MODEL.input_type == "colmap+completion":
                bxyz_input, _ = xyz_list_to_bxyz([xyz_voxel_list[i]])
                features_input = torch.ones(bxyz_input.shape[0], 1, device=bxyz_input.device, dtype=torch.float32)
                sparse_tensor_input = ME.SparseTensor(features=features_input, coordinates=bxyz_input)
                initializer_outputs = self.sgnn(sparse_tensor_input, gt_coords=bxyz_input)
                output_sparse = initializer_outputs["out"]
                xyz_voxel_init.append(output_sparse.C[:, 1:].clone())

                valid_mask = (xyz_voxel_init[i] >= bbox_voxel_list[i][0, :]).all(dim=1) & (xyz_voxel_init[i] <= bbox_voxel_list[i][1, :]).all(dim=1)
                xyz_voxel_init[i] = xyz_voxel_init[i][valid_mask]

            if not test_mode and xyz_voxel_init[i].shape[0] > self.config.DATASET.max_num_points:
                indices = torch.randperm(xyz_voxel_init[i].shape[0])[:self.config.DATASET.max_num_points]
                xyz_voxel_init[i] = xyz_voxel_init[i][indices]
                if rgb_init[i] is not None:
                    rgb_init[i] = rgb_init[i][indices]

            if test_mode and xyz_voxel_init[i].shape[0] > self.config.DATASET.eval_max_num_points:
                indices = torch.randperm(xyz_voxel_init[i].shape[0])[:self.config.DATASET.eval_max_num_points]
                xyz_voxel_init[i] = xyz_voxel_init[i][indices]
                if rgb_init[i] is not None:
                    rgb_init[i] = rgb_init[i][indices]

            latent_init = self._build_init_latents(
                xyz_voxel_orig=xyz_voxel_list[i],
                rgb_orig=rgb_list[i],
                xyz_voxel_init=xyz_voxel_init[i],
                rgb_init=rgb_init[i],
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
        rgb_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.config.MODEL.SCAFFOLD.unit_scale, "unit_scale must be True in _build_init_latents"
        N = xyz_voxel_init.shape[0]
        device = xyz_voxel_init.device

        if rgb_init is None:
            # Get the output color based on KNN
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
        else:
            rgb = rgb_init

        opacity = torch.ones(N, 1, device=device) * self.config.MODEL.GSPLAT.init_opacity
        opacity = inverse_sigmoid(opacity)

        scale = torch.ones(N, 3, device=device) * self.config.MODEL.SCAFFOLD.unit_scale_multiplier * self.config.MODEL.SCAFFOLD.voxel_size
        scale = torch.clamp(scale, min=1e-8, max=1e2)
        scale = scale / self.config.MODEL.SCAFFOLD.voxel_size
        scale = inverse_softplus(scale)

        rotation = torch.zeros(N, 4, device=device)
        rotation[:, 0] = 1.0

        xyz_offsets = torch.zeros(N, 3, device=rgb.device)
        latent = torch.cat([xyz_offsets, scale, rotation, opacity, rgb], dim=1)
        zero_latent = torch.zeros(
            rgb.shape[0],
            self.config.MODEL.SCAFFOLD.hidden_dim - latent.shape[1],
            device=rgb.device,
        )
        latent = torch.cat([latent, zero_latent], dim=1)
        assert latent.ndim == 2
        assert latent.shape[1] == self.config.MODEL.SCAFFOLD.hidden_dim, f"{latent.shape[1]} != {self.config.MODEL.SCAFFOLD.hidden_dim}"
        return latent

    def train_step_geometry_only(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        assert self.num_inner_steps == 1, "Phase-1 training only supports num_inner_steps == 1"
        other_stuff = {}        # For debugging purposes
        loss_dict = {}
        metrics_dict = {}

        xyz_input = point_batch["xyz_voxel"]
        rgb_input = point_batch["rgb"]

        xyz_gt = point_batch["xyz_voxel_gt"]
        normal_gt = point_batch["normal_gt"]

        bxyz, _ = xyz_list_to_bxyz(xyz_input)
        bxyz_gt, _ = xyz_list_to_bxyz(xyz_gt)

        sparse_tensor = ME.SparseTensor(
            features=torch.cat(rgb_input, dim=0),
            coordinates=bxyz,
        )

        model_outputs = self._model(
            sparse_tensor,
            gt_coords=bxyz_gt,
            max_samples_second_last=self.config.DATASET.max_num_points * self.config.TRAIN.batch_size // 8,
            max_gt_samples_last=self.config.DATASET.max_num_points * self.config.TRAIN.batch_size // 4,
            # verbose=self.config.debug,
        )

        output_sparse = model_outputs["out"]
        targets = model_outputs["targets"]
        out_cls = model_outputs["out_cls"]
        indices_gt = model_outputs["indices_gt"]
        indices_pred = model_outputs["indices_pred"]

        occ_loss = self.compute_occ_loss(out_cls, targets)
        loss_dict["occ_loss"] = occ_loss * self.config.MODEL.OPT.occ_mult

        xyz_densified, params_densified = output_sparse.decomposed_coordinates_and_features
        # Make sure that the rotation is aligned with the normal
        if self.config.MODEL.OPT.normal_loss_mult > 0 and "normal" in self.config.MODEL.INIT.init_type:
            normal_gt = torch.cat(normal_gt, dim=0)
            rotation_pred = output_sparse.F[:, -4:]
            normal_pred = quaternion_to_normal(rotation_pred)

            normal_pred = normal_pred[indices_pred]
            normal_gt = normal_gt[indices_gt]
            # Ignore invalid GT normals
            mask = normal_gt.norm(dim=1) > 0.9
            normal_gt = normal_gt[mask]
            normal_pred = normal_pred[mask]
            # normal_loss = 1 - torch.sum(normal_pred * normal_gt, dim=1)
            normal_loss = 1 - torch.abs(torch.sum(normal_pred * normal_gt, dim=1))
            loss_dict["normal_loss"] = normal_loss.nan_to_num().mean() * self.config.MODEL.OPT.normal_loss_mult

        with torch.no_grad():
            chamfer_full = chamfer_dist(xyz_densified[0], xyz_gt[0], norm=1)
            metrics_dict["chamfer_full"] = chamfer_full.item()

        if self.config.debug:
            if step_idx % 5 == 0:
                print(f"Occ loss: {occ_loss.item():.4f}")
                print(f"Chamfer full: {chamfer_full.item():.4f}")
        return (
            loss_dict,
            metrics_dict,
            other_stuff,
        )

    def train_step_jointly(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        assert self.num_inner_steps == 1, "Phase-1 training only supports num_inner_steps == 1"
        # timestamp is not used
        # timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)
        batch_gpu = self.move_to_device(batch_cpu)

        other_stuff = {}        # For debugging purposes
        loss_dict = {}
        metrics_dict = {}

        bbox_voxel = point_batch["bbox_voxel"]
        xyz_input = point_batch["xyz_voxel"]
        rgb_input = point_batch["rgb"]

        xyz_gt = point_batch["xyz_voxel_gt"]
        normal_gt = point_batch["normal_gt"]

        bxyz, _ = xyz_list_to_bxyz(xyz_input)
        bxyz_gt, _ = xyz_list_to_bxyz(xyz_gt)

        sparse_tensor = ME.SparseTensor(
            features=torch.cat(rgb_input, dim=0),
            coordinates=bxyz,
        )

        model_outputs = self._model(
            sparse_tensor,
            gt_coords=bxyz_gt,
            max_samples_second_last=self.config.DATASET.max_num_points * self.config.TRAIN.batch_size // 8,
            max_gt_samples_last=self.config.DATASET.max_num_points * self.config.TRAIN.batch_size // 4,
            # verbose=self.config.debug,
        )
        output_sparse = model_outputs["out"]
        targets = model_outputs["targets"]
        out_cls = model_outputs["out_cls"]
        occ_logits_last = model_outputs["last_prob"]
        indices_gt = model_outputs["indices_gt"]
        indices_pred = model_outputs["indices_pred"]

        occ_loss = self.compute_occ_loss(out_cls, targets)
        loss_dict["occ_loss"] = occ_loss * self.config.MODEL.OPT.occ_mult

        xyz_densified, params_densified = output_sparse.decomposed_coordinates_and_features
        # Make sure that the rotation is aligned with the normal
        if self.config.MODEL.OPT.normal_loss_mult > 0 and "normal" in self.config.MODEL.INIT.init_type:
            normal_gt = torch.cat(normal_gt, dim=0)
            rotation_pred = output_sparse.F[:, -4:]
            normal_pred = quaternion_to_normal(rotation_pred)

            normal_pred = normal_pred[indices_pred]
            normal_gt = normal_gt[indices_gt]
            # Ignore invalid GT normals
            mask = normal_gt.norm(dim=1) > 0.9
            normal_gt = normal_gt[mask]
            normal_pred = normal_pred[mask]
            # normal_loss = 1 - torch.sum(normal_pred * normal_gt, dim=1)
            normal_loss = 1 - torch.abs(torch.sum(normal_pred * normal_gt, dim=1))
            loss_dict["normal_loss"] = normal_loss.nan_to_num().mean() * self.config.MODEL.OPT.normal_loss_mult

        # Remove generated voxels that are outside of voxel_bbox and subsample if needed
        xyz_densified_subsample = []
        params_densified_subsample = []
        occ_densified_subsample = []

        for i in range(len(self.scaffolds)):
            # print("Before filter", xyz_densified[i].shape)
            valid_mask = (xyz_densified[i] >= bbox_voxel[i][0, :]).all(dim=1) & (xyz_densified[i] <= bbox_voxel[i][1, :]).all(dim=1)
            xyz_voxel = xyz_densified[i][valid_mask]
            occ = occ_logits_last.features_at(batch_index=i)[valid_mask]
            params = params_densified[i][valid_mask]

            if xyz_voxel.shape[0] > self.config.DATASET.max_num_points:
                indices = torch.randperm(xyz_voxel.shape[0])[:self.config.DATASET.max_num_points]
                xyz_voxel = xyz_voxel[indices]
                occ = occ[indices]
                params = params[indices]

            xyz_densified_subsample.append(xyz_voxel)
            occ_densified_subsample.append(occ)
            params_densified_subsample.append(params)

        with torch.no_grad():
            chamfer_full = chamfer_dist(xyz_densified_subsample[0], xyz_gt[0], norm=1)
            metrics_dict["chamfer_full"] = chamfer_full.item()

        if self.config.debug:
            if step_idx % 5 == 0:
                CONSOLE.log(f"Occ loss: {occ_loss.item():.4f}")
                CONSOLE.log(f"Chamfer full: {chamfer_full.item():.4f}")

        # Compose the scaffold using the new points
        # params_list is (N, 3 + 3 + 4) or (N, 2 + 3 + 4)
        latents_per_scene = []
        assert len(params_densified_subsample) == len(self.scaffolds)
        for i in range(len(params_densified_subsample)):
            if self.config.MODEL.gaussian_type == "3d":
                latents_per_scene.append(
                    self._build_latents(
                        scale=params_densified_subsample[i][:, 0:3],
                        opacity=occ_densified_subsample[i],
                        rgb=params_densified_subsample[i][:, 3:6],
                        rotation=params_densified_subsample[i][:, 6:],
                        voxel_to_world=self.scaffolds[i].transform,
                    )
                )
            else:
                latents_per_scene.append(
                    self._build_latents(
                        scale=params_densified_subsample[i][:, 0:2],
                        opacity=occ_densified_subsample[i],
                        rgb=params_densified_subsample[i][:, 2:5],
                        rotation=params_densified_subsample[i][:, 5:],
                        voxel_to_world=self.scaffolds[i].transform,
                    )
                )
            # Update the scaffold
            self.scaffolds[i].set_raw_params_and_xyz({"latent": latents_per_scene[i]}, xyz_densified_subsample[i])

        if self.config.MODEL.INIT.opt_enable:
            raise NotImplementedError("Optimizer model is not implemented in Phase-1 training")
            # grads: List[Dict[torch.Tensor]] = []
            # # grad_2d = []
            # # grad_2d_norm = []

            # for i in range(len(self.scaffolds)):
            #     params = self.scaffolds[i].get_raw_params()
            #     if self.config.MODEL.OPT.mask_gradients:
            #         grads.append({k: torch.zeros_like(v) for k, v in params.items()})
            #     else:
            #         grad_dict = self.get_params_grad(self.scaffolds[i], params, self.current_train_scene_ids[i])
            #         grads.append(grad_dict["grad_input"])
            #         # grad_2d.append(grad_dict["grad_2d"])
            #         # grad_2d_norm.append(grad_dict["grad_2d_norm"])

            # bxyz, input_sizes = xyz_list_to_bxyz(xyz_densified_subsample)
            # input_sizes = np.cumsum([0] + input_sizes)

            # params_cat = {"latent": torch.cat(latents_per_scene, dim=0)}
            # grads_cat = {"latent": torch.cat([g["latent"] for g in grads], dim=0)}

            # outputs = self._model(bxyz, params_cat, grads_cat, timestamp=timestamp)
            # latent_delta = (
            #     outputs["latent"]
            #     * self.config.MODEL.OPT.output_scale
            #     * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
            # )
            # params_cat["latent"] = params_cat["latent"] + latent_delta

            # latents_per_scene_new = []
            # for i in range(len(self.scaffolds)):
            #     latents_per_scene_new.append(params_cat["latent"][input_sizes[i]:input_sizes[i+1]])
        else:
            # Compute the loss without going through the optimizer model
            # The optimizer won't be learning anything in this case
            latents_per_scene_new = latents_per_scene

        scaffold_loss_dict, scaffold_metrics_dict, _gs_params = self.compute_scaffold_loss(
            latents_per_scene_new,
            batch_cpu,
            batch_gpu,
            xyz_voxels=xyz_densified_subsample,
        )
        loss_dict.update(scaffold_loss_dict)
        metrics_dict.update(scaffold_metrics_dict)

        if self.config.MODEL.use_identity_loss:
            identity_loss_dict, identity_metrics_dict = self.compute_identity_loss(
                latents_per_scene_new,
                batch_cpu,
                batch_gpu,
                xyz_voxels=xyz_densified_subsample,
            )
            loss_dict.update(identity_loss_dict)
            metrics_dict.update(identity_metrics_dict)

        return (
            loss_dict,
            metrics_dict,
            other_stuff,
        )

    def train_step(
        self,
        batch_cpu: Dict[str, torch.Tensor],
        point_batch: Dict[str, torch.Tensor],
        inner_step_idx: int,
        step_idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any]:
        if self.config.MODEL.INIT.train_geometry_only_steps > 0 and step_idx < self.config.MODEL.INIT.train_geometry_only_steps:
            return self.train_step_geometry_only(batch_cpu, point_batch, inner_step_idx, step_idx)
        else:
            return self.train_step_jointly(batch_cpu, point_batch, inner_step_idx, step_idx)

    def validate(self, step_idx: int, save_path: Optional[str] = None) -> Dict[str, float]:
        self.model.eval()
        self.scaffold_decoder.eval()

        metrics_list_init = []
        metrics_list_final = []
        for scene_idx, scene_id in enumerate(tqdm(self.val_dataset.scene_list, desc="Validation")):
            inner_step_idx = 0
            timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)

            point_data = self.load_scene_point_pair(scene_id)
            scaffold = self.init_scaffold_test(scene_id, test_mode=True)

            xyz_voxel = point_data["xyz_voxel"]
            rgb = point_data["rgb"]
            xyz_voxel_gt = point_data["xyz_voxel_gt"]
            normal_gt = point_data["normal_gt"]
            voxel_bbox = point_data["bbox_voxel"]

            bxyz, _ = xyz_list_to_bxyz([xyz_voxel])
            bxyz_gt, _ = xyz_list_to_bxyz([xyz_voxel_gt])

            sparse_tensor = ME.SparseTensor(
                features=rgb,
                coordinates=bxyz,
            )
            with torch.no_grad():
                model_outputs = self.model(
                    sparse_tensor,
                    gt_coords=bxyz_gt,
                )

            output_sparse = model_outputs["out"]
            targets = model_outputs["targets"]
            out_cls = model_outputs["out_cls"]
            occ_logits_last = model_outputs["last_prob"]
            indices_gt = model_outputs["indices_gt"]
            indices_pred = model_outputs["indices_pred"]
            occ_loss = self.compute_occ_loss(out_cls, targets)

            # Make sure that the rotation is aligned with the normal
            rotation_pred = output_sparse.F[:, -4:]
            normal_pred = quaternion_to_normal(rotation_pred)

            normal_pred_selected = normal_pred[indices_pred]
            normal_gt_selected = normal_gt[indices_gt]
            mask = normal_gt_selected.norm(dim=1) > 0.9
            normal_gt_selected = normal_gt_selected[mask]
            normal_pred_selected = normal_pred_selected[mask]
            normal_loss = 1 - torch.abs(torch.sum(normal_pred_selected * normal_gt_selected, dim=1)).nan_to_num()
            normal_loss = normal_loss.mean()

            xyz_densified, params_densified = output_sparse.decomposed_coordinates_and_features
            xyz_densified = xyz_densified[0]
            params_densified = params_densified[0]
            valid_mask = (xyz_densified >= voxel_bbox[0, :]).all(dim=1) & (xyz_densified <= voxel_bbox[1, :]).all(dim=1)
            xyz_densified = xyz_densified[valid_mask]
            occ_densified = occ_logits_last.features_at(batch_index=0)[valid_mask]
            params_densified = params_densified[valid_mask]
            normal_pred = normal_pred[valid_mask]
            if xyz_densified.shape[0] > self.config.DATASET.eval_max_num_points:
                indices = torch.randperm(xyz_densified.shape[0])[:self.config.DATASET.eval_max_num_points]
                xyz_densified = xyz_densified[indices]
                occ_densified = occ_densified[indices]
                params_densified = params_densified[indices]
                normal_pred = normal_pred[indices]

            # Evaluate Chamfer distance between predicted and GT points
            chamfer_full = chamfer_dist(xyz_densified, xyz_voxel_gt, norm=1).item()
            chamfer_pred_to_gt = chamfer_dist(xyz_densified, xyz_voxel_gt, norm=1, single_directional=True).item()
            chamfer_gt_to_pred = chamfer_dist(xyz_voxel_gt, xyz_densified, norm=1, single_directional=True).item()

            if self.config.MODEL.gaussian_type == "3d":
                latents_new = self._build_latents(
                    scale=params_densified[:, 0:3],
                    opacity=occ_densified,
                    rgb=params_densified[:, 3:6],
                    rotation=params_densified[:, 6:],
                    voxel_to_world=point_data["voxel_to_world"],
                )
            else:
                latents_new = self._build_latents(
                    scale=params_densified[:, 0:2],
                    opacity=occ_densified,
                    rgb=params_densified[:, 2:5],
                    rotation=params_densified[:, 5:],
                    voxel_to_world=point_data["voxel_to_world"],
                )
            scaffold.set_raw_params_and_xyz({"latent": latents_new}, xyz_densified)

            init_metrics = self.evaluate_gs(
                scaffold=scaffold,
                scene_id=scene_id,
                save_path=save_path,
                save_file_suffix="init",
                use_identity=True,
            )
            metrics_list_init.extend(init_metrics)

            if self.config.MODEL.INIT.opt_enable:
                raise NotImplementedError("Optimizer model is not implemented in Phase-1 training")
                params = scaffold.get_raw_params()
                grad_dict = self.get_params_grad(scaffold, params, scene_id, fixed=True)
                grads = grad_dict["grad_input"]

                bxyz, _ = xyz_list_to_bxyz([scaffold.xyz_voxel])
                with torch.no_grad():
                    outputs = self._model(bxyz, params, grads, timestamp=timestamp)
                    latent_delta = (
                        outputs["latent"]
                        * self.config.MODEL.OPT.output_scale
                        * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
                    )
                    params["latent"] = params["latent"] + latent_delta

                scaffold.set_raw_params(params)

                final_metrics = self.evaluate_gs(
                    scaffold=scaffold,
                    scene_id=scene_id,
                    save_path=save_path,
                    save_file_suffix="final",
                    use_identity=False,
                )
            else:
                final_metrics = deepcopy(init_metrics)

            for x in final_metrics:
                # Copy the other metrics
                x["chamfer_full"] = chamfer_full
                x["chamfer_pred_to_gt"] = chamfer_pred_to_gt
                x["chamfer_gt_to_pred"] = chamfer_gt_to_pred
                x["occ_loss"] = occ_loss.item()
                x["normal_loss"] = normal_loss.item()

            metrics_list_final.extend(final_metrics)

        final_metrics = {}
        for k, v in metrics_list_final[0].items():
            final_metrics[k] = np.mean([m[k] for m in metrics_list_final])

        for k in metrics_list_init[0].keys():
            final_metrics[f"{k}_init"] = np.mean([m[k] for m in metrics_list_init])
        self.model.train()
        self.scaffold_decoder.train()
        CONSOLE.print(json.dumps(final_metrics, indent=4))
        return final_metrics
