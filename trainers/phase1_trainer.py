from typing import Any, Union, Optional, List, Tuple, Dict, Literal
from pathlib import Path

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from trainers.base_trainer import BaseTrainer
from trainers.quicksplat_base_trainer import QuickSplatTrainer
from models.optimizer import UNetOptimizer
from models.initializer import Initializer18
from models.unet_base import ResUNetConfig
from modules.rasterizer_3d import ScaffoldRasterizer, Camera
from modules.rasterizer_2d import Scaffold2DGSRasterizer

from utils.pose import apply_transform, apply_rotation, quaternion_to_normal
from utils.optimizer import Optimizer
from utils.sparse import xyz_list_to_bxyz, chamfer_dist, chamfer_dist_with_crop


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
        return Initializer18(
            in_channels=3,
            out_channels=out_channels,
            config=ResUNetConfig(),
            use_time_emb=False,
            dense_bottleneck=True,
            num_dense_blocks=self.config.MODEL.INIT.num_dense_blocks,
        )

    def setup_optimizers(self):
        # param_dict = self.model.get_param_groups()
        param_dict = {
            "model": self.model.parameters(),
        }
        self.optimizer = Optimizer(param_dict, self.config.OPTIMIZER)

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
            verbose=self.config.debug,
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
            rotation_pred = output_sparse.F[:, 6:10]
            normal_pred = quaternion_to_normal(rotation_pred)

            normal_pred = normal_pred[indices_pred]
            normal_gt = normal_gt[indices_gt]
            # Ignore invalid normals
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

        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps)
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
            verbose=self.config.debug,
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
            rotation_pred = output_sparse.F[:, 6:10]
            normal_pred = quaternion_to_normal(rotation_pred)

            normal_pred = normal_pred[indices_pred]
            normal_gt = normal_gt[indices_gt]
            # Ignore invalid normals
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
                print(f"Occ loss: {occ_loss.item():.4f}")
                print(f"Chamfer full: {chamfer_full.item():.4f}")

        # Compose the scaffold using the new points
        # params_list is (N, 3 + 3 + 4)
        latents_per_scene = []
        assert len(params_densified_subsample) == len(self.scaffolds)
        for i in range(len(params_densified_subsample)):
            latents_per_scene.append(
                self._build_latents(
                    scale=params_densified_subsample[i][:, 0:3],
                    opacity=occ_densified_subsample[i],
                    rgb=params_densified_subsample[i][:, 3:6],
                    rotation=params_densified_subsample[i][:, 6:],
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
