from typing import Any, Union, Optional, List, Tuple, Dict, Literal
from pathlib import Path
import functools
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import MinkowskiEngine as ME
import pytorch3d.transforms as transforms3d

from trainers.base_trainer import BaseTrainer
from models.optimizer import UNetOptimizer
from models.initializer import Initializer18
from models.unet_base import ResUNetConfig
from models.scaffold_gs import (
    ScaffoldGSFull,
    GSIdentityDecoder3D,
    GSIdentityDecoder2D,
    GSDecoder3D,
    GSDecoder2D,
    inverse_sigmoid,
    inverse_softplus,
)
from modules.rasterizer_3d import Camera

from utils.depth import save_depth, depth_loss, log_depth_loss, compute_full_depth_metrics, save_depth_opencv
from utils.pose import apply_transform, apply_rotation, quaternion_to_normal
from utils.fusion import MeshExtractor
from utils.utils import TimerContext


class QuickSplatTrainer(BaseTrainer):
    def setup_dataloader(self):
        # TODO
        raise NotImplementedError("TODO")

    def get_inner_dataset(self, scene_id, fixed=False):
        # TODO
        inner_train_dataset = ScannetppDataset(
            self.config.DATASET.source_path,
            self.ply_path,
            scene_id=scene_id,
            split="train",
            # downsample=self.config.MODEL.OPT.inner_downsample,
            downsample=self.config.DATASET.image_downsample,
            num_train_frames=self.config.DATASET.num_train_frames,
            subsample_randomness=not fixed,
        )

        inner_train_loader = DataLoader(
            inner_train_dataset,
            batch_size=self.config.MODEL.OPT.inner_batch_size,
            shuffle=True,
            num_workers=self.config.TRAIN.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return inner_train_dataset, inner_train_loader

    def get_identity_decoder(self):
        assert self.config.MODEL.gaussian_type in ["2d", "3d"], f"Unknown gaussian type {self.config.MODEL.gaussian_type}"
        if self.config.MODEL.gaussian_type == "3d":
            decoder_class = GSIdentityDecoder3D
        elif self.config.MODEL.gaussian_type == "2d":
            decoder_class = GSIdentityDecoder2D
        else:
            raise ValueError(f"Unknown gaussian type {self.config.MODEL.gaussian_type}")
        return decoder_class(
            exp_scale=self.config.MODEL.SCAFFOLD.exp_scale,
            max_scale=self.config.MODEL.SCAFFOLD.max_scale,
            scale_activation=self.config.MODEL.SCAFFOLD.scale_activation,
            offset_scaling=self.config.MODEL.SCAFFOLD.offset_scaling,
        ).to(self.device)

    def get_scaffold_decoder(self):
        assert self.config.MODEL.gaussian_type in ["2d", "3d"], f"Unknown gaussian type {self.config.MODEL.gaussian_type}"
        if self.config.MODEL.gaussian_type == "3d":
            decoder_class = GSDecoder3D
        elif self.config.MODEL.gaussian_type == "2d":
            decoder_class = GSDecoder2D

        return decoder_class(
            input_dim=self.config.MODEL.SCAFFOLD.hidden_dim,
            hidden_dim=self.config.MODEL.SCAFFOLD.hidden_dim * 2,
            num_gs=self.config.MODEL.SCAFFOLD.num_gs,
            num_layers=self.config.MODEL.SCAFFOLD.num_layers,
            skip_connect=self.config.MODEL.SCAFFOLD.skip_connect,
            exp_scale=self.config.MODEL.SCAFFOLD.exp_scale,
            scale_activation=self.config.MODEL.SCAFFOLD.scale_activation,
            max_scale=self.config.MODEL.SCAFFOLD.max_scale,
            quat_rotation=self.config.MODEL.SCAFFOLD.quat_rotation,
            normal_zero_init=self.config.MODEL.SCAFFOLD.normal_zero_init,
            modify_xyz_offsets=self.config.MODEL.SCAFFOLD.modify_xyz_offsets,
            offset_scaling=self.config.MODEL.SCAFFOLD.offset_scaling,
        ).to(self.device)

    def setup_decoders(self):
        # Options: 2DGS, 3DGS, 2DIdentity, 3DIdentity
        if self.config.MODEL.decoder_type == "identity":
            scaffold_decoder = self.get_identity_decoder()
        elif self.config.MODEL.decoder_type == "scaffold":
            scaffold_decoder = self.get_scaffold_decoder()
        if self.world_size > 1:
            self.scaffold_decoder = DDP(scaffold_decoder, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            self.scaffold_decoder = scaffold_decoder

        self.identity_decoder = self.get_identity_decoder()

    def _get_timestamp(
        self,
        inner_step_idx: int,
        num_inner_steps: int,
        max_steps_train: Optional[int] = None,
    ):
        MAX_STEP = 100
        if self.config.MODEL.OPT.timestamp_random:
            start_idx = (MAX_STEP // self.config.MODEL.OPT.num_steps) * inner_step_idx
            end_idx = (MAX_STEP // self.config.MODEL.OPT.num_steps) * (inner_step_idx + 1)
            timestamp = np.random.randint(start_idx, end_idx) * 1.0 / MAX_STEP
        else:
            if self.config.MODEL.OPT.timestamp_norm:
                timestamp = (inner_step_idx + 1.0) / num_inner_steps
            else:
                timestamp = inner_step_idx + 1
                if max_steps_train is not None:
                    timestamp = min(timestamp, max_steps_train)
        return timestamp

    def compute_occ_loss(
        self,
        out_cls,
        targets,
        ignores: Optional[List[torch.Tensor]] = None,
    ):
        # Compute occ loss
        occ_loss = 0.0
        assert self.config.MODEL.OPT.occ_mult > 0
        for idx, (pred, gt) in enumerate(zip(out_cls, targets)):
            pred = pred.F.flatten(0, 1)
            gt = gt.float()

            if self.config.MODEL.DENSIFIER.loss_ignore_exist:
                assert ignores is not None
                ignore_mask = ignores[idx]
                # Ignore the existing points in "bxyz" in each level
                # Could be a problem where most of the voxels are ignored
                valid_mask = ~ignore_mask
                if self.config.debug:
                    ratio = valid_mask.sum() / valid_mask.numel()
                    print(f"{idx}: valid: {valid_mask.sum()}/{valid_mask.numel()}, ratio: {ratio:.2f}")
                pred = pred[valid_mask]
                gt = gt[valid_mask]

            if self.config.MODEL.INIT.use_balance_weight:
                num_pos = gt.sum()
                num_neg = gt.numel() - num_pos
                pos_weight = num_neg / num_pos
                pos_weight = pos_weight.detach()
                if self.config.debug:
                    print(f"{idx}: shape: {gt.shape}, weight: {pos_weight:.2f}")
            else:
                pos_weight = None
            occ = nn.functional.binary_cross_entropy_with_logits(
                pred,
                gt,
                pos_weight=pos_weight,
                # weight=loss_mask,
            )

            occ_loss += occ / len(targets)
        return occ_loss

    def train(self):
        """Basic training loop:
        for each iteration:
            Load point cloud batch. Init the scaffolds for each scene in the batch by
            self.init_scaffold_train_batch()
            for each inner step:
                self.train_step()
            self.after_train_step()
        """
        self._model.train()
        self.best_psnr = 0.0
        self.num_inner_steps = 1    # Gradually increase during the training

        num_iterations = self.config.TRAIN.num_iterations
        if self.world_size > 1:
            dist.barrier()

        pbar = tqdm(total=num_iterations, desc="Training")
        if self._start_step > 1:
            pbar.update(self._start_step - 1)

        for step_idx in range(self._start_step, num_iterations + 1):
            self.iter_timer.start()
            pbar.update(1)

            try:
                batch = next(self.train_iter)
                point_batch = next(self.train_point_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
                self.train_point_iter = iter(self.train_point_loader)
                point_batch = next(self.train_point_iter)

            scene_ids = batch["scene_id"][0]
            # assert scene_ids == point_batch["scene_id"]
            # for key, x in batch.items():
            #     if isinstance(x, torch.Tensor):
            #         print(key, x.shape)
            # print(batch["depth"].shape, batch["depth"].min(), batch["depth"].max())

            # Reset scaffold
            point_batch = self.move_to_device(point_batch)
            self.scaffolds: List[ScaffoldGSFull] = self.init_scaffold_train_batch(
                point_batch["xyz"],
                point_batch["rgb"],
                point_batch["xyz_voxel"],
                point_batch["transform"],
                point_batch["bbox"],
                point_batch["bbox_voxel"],
                test_mode=False,
            )
            self.current_train_scene_ids = scene_ids

            if self.config.MODEL.model_type == "with_memory" or self.config.MODEL.model_type == "with_feature":
                assert not self.model.has_memory()

            for inner_step_idx in range(self.num_inner_steps):

                loss_dict, metrics_dict, other_stuff = self.train_step(
                    batch, point_batch, inner_step_idx, step_idx,
                )
                loss = functools.reduce(torch.add, loss_dict.values())
                loss_dict["total_loss"] = loss
                loss.backward()

                self.before_optimizer(step_idx)

                norm_dict = self.optimizer.get_max_norm()

                if self.config.MODEL.OPT.decoder_update_last_only:
                    raise NotImplementedError("TODO: Remove")
                    # if inner_step_idx != self.num_inner_steps - 1:
                    #     # Don't update the scaffold decoder unless it's the last step
                    #     self.optimizer.zero_grad_target("decoder")

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Shouldn't encounter StopIteration here
                if inner_step_idx < self.num_inner_steps - 1:
                    batch = next(self.train_iter)

            self.after_train_step(step_idx)

            metrics_dict["num_inner_steps"] = self.num_inner_steps
            for key, val in norm_dict.items():
                metrics_dict[f"grad_norm_{key}"] = val

            self.train_metrics.update(metrics_dict)
            self.train_metrics.update(loss_dict)
            self.iter_timer.stop()

            if step_idx % self.config.TRAIN.log_interval == 0:
                self.log_metrics(step_idx)

            if step_idx % self.config.TRAIN.val_interval == 0:
                save_dir = self.output_dir / "val"
                # Check main thread
                if self.local_rank == 0:
                    val_metrics = self.validate(step_idx, save_path=save_dir)
                    self.write_scalars("val", val_metrics, step_idx)

                    if self.config.save_checkpoint:
                        self.save_checkpoint(step_idx, val_metrics)
                else:
                    val_metrics = self.validate(step_idx, save_path=None)

            self.optimizer.step_scheduler()

            if self.world_size > 1:
                dist.barrier()
        pbar.close()

    def compute_outer_loss(
        self,
        images_pred: torch.Tensor,
        images_gt: torch.Tensor,
        gs_params: List[Dict[str, torch.Tensor]],
        distortion: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        normal_from_depth: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        depth_gt: Optional[torch.Tensor] = None,
    ):
        loss_dict = {}
        assert images_gt.shape == images_pred.shape
        if images_gt.ndim == 3:
            # (3, H, W) -> (1, 3, H, W)
            images_gt = images_gt.unsqueeze(0)
            images_pred = images_pred.unsqueeze(0)

        loss_dict["l1"] = self.l1(images_pred, images_gt)
        if self.config.MODEL.OPT.ssim_mult > 0:
            ssim_loss = 1 - self.ssim(images_pred, images_gt)
            loss_dict["ssim_loss"] = ssim_loss * self.config.MODEL.OPT.ssim_mult

        if self.config.MODEL.OPT.lpips_mult > 0:
            loss_dict["lpips_loss"] = self.lpips(
                torch.clamp(images_pred, 0, 1),
                images_gt,
            ) * self.config.MODEL.OPT.lpips_mult

        if self.config.MODEL.OPT.vgg_mult > 0:
            loss_dict["vgg_loss"] = self.vgg(
                torch.clamp(images_pred, 0, 1),
                images_gt,
            ) * self.config.MODEL.OPT.vgg_mult

        if self.config.MODEL.OPT.dist_mult > 0:
            assert distortion is not None
            loss_dict["dist_loss"] = distortion.mean() * self.config.MODEL.OPT.dist_mult

        if self.config.MODEL.OPT.normal_mult > 0:
            assert normal is not None
            assert normal_from_depth is not None
            normal_error = (1 - (normal * normal_from_depth).sum(dim=1))
            loss_dict["normal_loss"] = normal_error.mean() * self.config.MODEL.OPT.normal_mult

        assert len(gs_params) == len(self.scaffolds)
        # Regularization losses
        if self.config.MODEL.scale_2d_reg_mult > 0:
            loss_dict["scale_2d_reg_loss"] = 0.0
            # Eq 7 in paper: \sum_i \max (0, min(scale) - \eps)
            for i in range(len(self.scaffolds)):
                scale = gs_params[i]["scale"]
                scale_with_min_axis = torch.min(scale, dim=1).values
                reg_loss = torch.clamp(scale_with_min_axis - 1e-2, min=0) / len(self.scaffolds)
                loss_dict["scale_2d_reg_loss"] += torch.mean(reg_loss) * self.config.MODEL.scale_2d_reg_mult

        if self.config.MODEL.scale_3d_reg_mult > 0:
            loss_dict["scale_3d_reg_loss"] = 0.0
            for i in range(len(self.scaffolds)):
                scale = gs_params[i]["scale"]
                reg_loss = torch.mean(torch.norm(scale, p=2, dim=1)) / len(self.scaffolds)
                loss_dict["scale_3d_reg_loss"] += reg_loss * self.config.MODEL.scale_3d_reg_mult

        if self.config.MODEL.opacity_reg_mult > 0:
            loss_dict["opacity_reg_loss"] = 0.0
            for i in range(len(self.scaffolds)):
                opacity = gs_params[i]["opacity"]
                reg_loss = torch.mean(opacity) / len(self.scaffolds)
                loss_dict["opacity_reg_loss"] += reg_loss * self.config.MODEL.opacity_reg_mult

        if self.config.MODEL.OPT.depth_mult > 0:
            assert depth is not None
            assert depth_gt is not None
            # print(depth.shape, depth.min(), depth.max())
            # print("GT", depth_gt.shape, depth_gt.min(), depth_gt.max())
            loss_dict["depth_loss"] = depth_loss(depth, depth_gt) * self.config.MODEL.OPT.depth_mult

        if self.config.MODEL.OPT.log_depth_mult > 0:
            assert depth is not None
            assert depth_gt is not None
            loss_dict["log_depth_loss"] = log_depth_loss(depth, depth_gt) * self.config.MODEL.OPT.log_depth_mult

        return loss_dict

    def get_params_grad(
        self,
        scaffold: ScaffoldGSFull,
        params: Dict[str, torch.Tensor],
        scene_id: str,
        fixed: bool = False,
    ):
        file_names = []
        with TimerContext("prepare get_params_grad", self.config.debug):
            param_keys = list(params.keys())
            assert param_keys == ["latent"]
            grads = {key: torch.zeros_like(params[key]) for key in param_keys}
            if self.config.MODEL.decoder_type == "identity":
                num_gs = 1
            else:
                num_gs = self.config.MODEL.SCAFFOLD.num_gs
            grad_2d = torch.zeros(scaffold.num_points * num_gs, 2, device=self.device)
            grad_2d_norm = torch.zeros(scaffold.num_points * num_gs, 1, device=self.device)
            counter = torch.zeros(scaffold.num_points * num_gs, 1, device=self.device)

            _, inner_loader = self.get_inner_dataset(scene_id, fixed=fixed)

            if self.world_size > 1:
                gs_params = self.scaffold_decoder.module(
                    latent=params["latent"],
                    xyz_voxel=scaffold.xyz_voxel,
                    transform=scaffold.transform,
                    voxel_size=scaffold.voxel_size,
                )
            else:
                gs_params = self.scaffold_decoder(
                    latent=params["latent"],
                    xyz_voxel=scaffold.xyz_voxel,
                    transform=scaffold.transform,
                    voxel_size=scaffold.voxel_size,
                )
            # Reshape the gs attributes from (N, M, C) to (N*M, C)
            for key in gs_params:
                gs_params[key] = gs_params[key].flatten(0, 1).contiguous()
                # print(key, gs_params[key].shape)

        # TODO: Figure out using which as xyz_world
        xyz_world = gs_params["xyz"]
        rgb = gs_params["rgb"]
        # save_ply("xyz_world1.ply", xyz_world.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        xyz_world = apply_transform(scaffold.xyz_voxel.float(), scaffold.transform)
        # save_ply("xyz_world2.ply", xyz_world.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        features_3d = None
        point_weights = torch.zeros(xyz_world.shape[0], device=xyz_world.device)

        with TimerContext("run get_params_grad", self.config.debug):
            for i, x_cpu in enumerate(inner_loader):
                x_gpu = self.move_to_device(x_cpu)

                with TimerContext("get_params_grad forward (inner loop)", self.config.debug):
                    batch_size = x_cpu["rgb"].shape[0]
                    images_pred = []
                    normals = []
                    normals_from_depth = []
                    distortions = []
                    means_2d = []
                    vis_masks = []

                    for j in range(batch_size):
                        camera = Camera(
                            fov_x=float(x_cpu["fov_x"][j]),
                            fov_y=float(x_cpu["fov_y"][j]),
                            height=int(x_cpu["height"][j]),
                            width=int(x_cpu["width"][j]),
                            image=x_gpu["rgb"][j],
                            world_view_transform=x_gpu["world_view_transform"][j],
                            full_proj_transform=x_gpu["full_proj_transform"][j],
                            camera_center=x_gpu["camera_center"][j],
                        )

                        render_outputs = self.rasterizer(
                            gs_params,
                            camera,
                            active_sh_degree=0,     # Does not matter
                        )
                        file_names.append(x_cpu["file_name"][j])
                        images_pred.append(render_outputs["render"].unsqueeze(0))
                        vis_masks.append((render_outputs["radii"] > 0).unsqueeze(0))
                        means_2d.append(render_outputs["viewspace_points"])
                        counter[render_outputs["radii"] > 0] += 1

                        if "normal" in render_outputs:
                            normals.append(render_outputs["normal"])
                        if "normal_from_depth" in render_outputs:
                            normals_from_depth.append(render_outputs["normal_from_depth"])
                        if "distortion" in render_outputs:
                            distortions.append(render_outputs["distortion"])

                    # counter[valid_gs, :] += 1
                    images_pred = torch.cat(images_pred, dim=0)
                    vis_masks = torch.cat(vis_masks, dim=0)
                    # print(images_pred.shape)
                    images_gt = x_gpu["rgb"]

                    if self.config.MODEL.gaussian_type == "2d":
                        normals = torch.stack(normals, dim=0)
                        normals_from_depth = torch.stack(normals_from_depth, dim=0)
                        distortions = torch.stack(distortions, dim=0)
                    else:
                        normals = None
                        normals_from_depth = None
                        distortions = None

                    inner_loss_dict = self.compute_inner_loss(
                        images_pred,
                        images_gt,
                        distortion=distortions,
                        normal=normals,
                        normal_from_depth=normals_from_depth,
                    )
                    loss = functools.reduce(torch.add, inner_loss_dict.values())

                with TimerContext("get_params_grad backward (inner loop)", self.config.debug):
                    grad_batch = torch.autograd.grad(
                        loss,
                        [params[key] for key in param_keys] + means_2d,
                        retain_graph=True,
                        # allow_unused=True,
                    )
                grad_2d_batch = grad_batch[1:]
                grad_2d += sum(
                    grad_2d_batch[i][:, :2] for i in range(len(grad_2d_batch))
                )
                grad_2d_norm += sum(
                    grad_2d_batch[i][:, :2].norm(p=2, dim=1, keepdim=True) for i in range(len(grad_2d_batch))
                )

                for j, key in enumerate(param_keys):
                    # print(i, key, grad_batch[i].shape)
                    grads[key] += (
                        grad_batch[j]
                        .nan_to_num(0)
                        .clamp(-self.config.MODEL.OPT.max_grad, self.config.MODEL.OPT.max_grad)
                        .detach()
                    )
                del grad_batch
                del loss

                if self.config.MODEL.model_type == "with_feature":
                    if i == 0:
                        features_3d_batch = self.model.extract_and_lift_features(
                            xyz_world,
                            images_gt,
                            x_gpu["intrinsic"],
                            x_gpu["world_to_camera"],
                            normalize=True,
                        )   # (num_images, num_points, C)

                        features_3d = (features_3d_batch * vis_masks.unsqueeze(-1)).sum(dim=0)    # (num_points, C)
                    else:
                        with torch.no_grad():
                            features_3d_batch = self.model.extract_and_lift_features(
                                xyz_world,
                                images_gt,
                                x_gpu["intrinsic"],
                                x_gpu["world_to_camera"],
                                normalize=True,
                            )
                        features_3d += (features_3d_batch * vis_masks.unsqueeze(-1)).sum(dim=0)   # (num_points, C)
                    point_weights += vis_masks.sum(dim=0)

            for key in grads.keys():
                # normalize the channels by the largest values (inf-norm)
                denom = grads[key].norm(p=torch.inf, dim=0, keepdim=True).clamp_min(1e-12)
                grads[key] = grads[key] / denom

            grad_2d = grad_2d / torch.clamp_min(counter, 1)
            grad_2d_norm = grad_2d_norm / torch.clamp_min(counter, 1)

            if features_3d is not None:
                features_3d = features_3d / point_weights.clamp_min(1e-12).unsqueeze(-1)
                grads["latent"] = torch.cat([grads["latent"], features_3d], dim=-1)

            grad_2d = grad_2d.view(-1, num_gs, 2)
            grad_2d = torch.mean(grad_2d, dim=1)

            grad_2d_norm = grad_2d_norm.view(-1, num_gs, 1)
            grad_2d_norm = torch.mean(grad_2d_norm, dim=1)
            # SUM
            return {
                "grad_input": grads,
                "grad_2d": grad_2d,
                "grad_2d_norm": grad_2d_norm,
                # "features_3d": features_3d,
                "file_names": file_names,
            }

    def compute_inner_loss(
        self,
        images_pred: torch.Tensor,
        images_gt: torch.Tensor,
        distortion: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        normal_from_depth: Optional[torch.Tensor] = None,
    ):
        """
        This is used for computing the gradient for the optimizer's input
        """
        loss_dict = {}
        assert images_gt.shape == images_pred.shape
        if images_gt.ndim == 3:
            # (3, H, W) -> (1, 3, H, W)
            images_gt = images_gt.unsqueeze(0)
            images_pred = images_pred.unsqueeze(0)

        loss_dict["l1"] = self.l1(images_pred, images_gt)
        ssim_loss = 1 - self.ssim(images_pred, images_gt)
        loss_dict["ssim_loss"] = ssim_loss * self.config.MODEL.GSPLAT.lambda_dssim

        # Use 2DGS loss if needed
        if distortion is not None:
            loss_dict["dist_loss"] = distortion.mean() * self.config.MODEL.OPT.dist_mult

        if normal is not None and normal_from_depth is not None:
            normal_error = (1 - (normal * normal_from_depth).sum(dim=1))
            loss_dict["normal_loss"] = normal_error.mean() * self.config.MODEL.OPT.normal_mult

        return loss_dict

    def compute_scaffold_loss(
        self,
        latents: List[torch.Tensor],
        batch_cpu: Dict[str, torch.Tensor],
        batch_gpu: Dict[str, torch.Tensor],
        xyz_voxels: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Compute the major losses to train the intializer and the optimizer.
        It will decode the gs parameters from the latents and render the images.
        """

        loss_dict = {}
        metrics_dict = {}

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

        gs_params = [None for _ in range(batch_size)]
        for i in range(len(self.scaffolds)):
            if xyz_voxels is not None:
                xyz_voxel = xyz_voxels[i]
            else:
                xyz_voxel = self.scaffolds[i].xyz_voxel

            gs_params[i] = self.scaffold_decoder(
                latent=latents[i],
                xyz_voxel=xyz_voxel,
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
            metrics_dict["num_points"] = np.mean([x.shape[0] for x in latents])
            metrics_dict["num_gs"] = np.mean([x["xyz"].shape[0] for x in gs_params])
            metrics_dict["scale"] = torch.mean(torch.cat([x["scale"] for x in gs_params], dim=0))
            metrics_dict["opacity"] = torch.mean(torch.cat([x["opacity"] for x in gs_params], dim=0))

        return loss_dict, metrics_dict, gs_params

    def compute_identity_loss(
        self,
        latents: List[torch.Tensor],
        batch_cpu: Dict[str, torch.Tensor],
        batch_gpu: Dict[str, torch.Tensor],
        xyz_voxels: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute rendering loss function with the identity decoder. This will reduce the effect of the latents.
        """
        loss_dict = {}
        metrics_dict = {}

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
            # NOTE: Major fix here
            if xyz_voxels is not None:
                xyz_voxel = xyz_voxels[i]
            else:
                xyz_voxel = self.scaffolds[i].xyz_voxel
            gs_params[i] = self.identity_decoder(
                latent=latents[i],
                xyz_voxel=xyz_voxel,
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

        loss_dict = {f"identity_{k}": v for k, v in loss_dict.items()}
        metrics_dict = {f"identity_{k}": v for k, v in metrics_dict.items()}
        return loss_dict, metrics_dict

    def _build_latents(
        self,
        scale: torch.Tensor,
        opacity: torch.Tensor,
        rgb: torch.Tensor,
        rotation: torch.Tensor,
        voxel_to_world: torch.Tensor,
    ):
        if "rgb" not in self.config.MODEL.INIT.init_type:
            rgb = torch.zeros_like(rgb)

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

            # if not self.config.MODEL.DENSIFIER.init_rot_reverse:
            #     # TODO: Have to think about the order
            #     rotation = transforms3d.quaternion_multiply(rotation, voxel_to_world_rot)
            # else:
            #     # <- this is correct
            rotation = transforms3d.quaternion_multiply(voxel_to_world_rot, rotation)

        xyz_offsets = torch.zeros(rgb.shape[0], 3, device=rgb.device)
        zero_latent = torch.zeros(
            rgb.shape[0],
            (
                self.config.MODEL.SCAFFOLD.hidden_dim
                - xyz_offsets.shape[-1]
                - scale.shape[-1]
                - rotation.shape[-1]
                - opacity.shape[-1]
                - rgb.shape[-1]
            ),
            device=rgb.device,
        )

        return torch.cat([
            xyz_offsets,
            scale,
            rotation,
            opacity,
            rgb,
            zero_latent,
        ], dim=-1)
