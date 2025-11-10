from typing import Optional, List, Tuple, Dict, Literal
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import MinkowskiEngine as ME
import pytorch3d.transforms as transforms3d
from PIL import Image

from trainers.base_trainer import BaseTrainer
# TODO: Refactor the dataset
from dataset.scannetpp import ScannetppDataset
from dataset.scannetpp import MultiScannetppDataset
from dataset.scannetpp_points import MultiScannetppPointDataset
from dataset.scannetpp_points_aabb import MultiScannetppPointAABBDataset
from dataset.utils import RepeatSampler

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

from utils.rich_utils import CONSOLE
from utils.depth import depth_loss, log_depth_loss, compute_full_depth_metrics, save_depth_opencv
from utils.utils import TimerContext


class QuickSplatTrainer(BaseTrainer):
    def setup_dataloader(self):
        # Starting from 1 inner step
        self.num_inner_steps = 1

        if self.config.MODEL.input_type == "mesh":
            self.ply_path = self.config.DATASET.gt_ply_path
        else:
            self.ply_path = self.config.DATASET.ply_path

        use_depth_training = self.config.MODEL.OPT.depth_mult > 0 or self.config.MODEL.OPT.log_depth_mult > 0
        self.train_dataset = MultiScannetppDataset(
            self.config.DATASET.source_path,
            self.ply_path,
            self.config.DATASET.train_split_path,
            num_samples=self.config.DATASET.num_target_views,
            split="train",
            downsample=self.config.DATASET.image_downsample,
            num_train_frames=-1,    # Use all training frames for outer loop
            load_depth=use_depth_training,
        )

        self.train_point_dataset = MultiScannetppPointAABBDataset(
            self.config.DATASET.source_path,
            self.ply_path,
            self.config.DATASET.gt_ply_path,
            self.config.DATASET.train_split_path,
            self.config.DATASET.transform_path,
            crop_points=False if self.config.MODEL.input_type == "mesh" else True,
            # crop_points=False,
            noise_aug=self.config.DATASET.use_point_aug,
            global_aug=self.config.DATASET.use_point_rotate_aug,
            color_aug=self.config.DATASET.use_point_color_aug,
            voxelize_aug=self.config.DATASET.voxelize_aug,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
            max_points=self.config.DATASET.max_num_points,
            read_normal=False,
            return_gt=True,
            read_normal_gt=True,
        )

        assert self.train_dataset.scene_list == self.train_point_dataset.scene_list

        # To sync the data iterators
        self.data_generator = np.random.default_rng(self.config.MODEL.SCAFFOLD.seed + self.local_rank)
        self.data_point_generator = np.random.default_rng(self.config.MODEL.SCAFFOLD.seed + self.local_rank)

        sampler = RepeatSampler(
            self.train_dataset,
            batch_size=self.config.TRAIN.batch_size,
            num_repeat=self.num_inner_steps,
            # seed=self.config.MODEL.SCAFFOLD.seed,
            generator=self.data_generator,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.TRAIN.batch_size,
            num_workers=self.config.TRAIN.num_workers,
            # num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )

        sampler = RepeatSampler(
            self.train_point_dataset,
            batch_size=self.config.TRAIN.batch_size,
            num_repeat=1,
            generator=self.data_point_generator,
        )

        self.train_point_loader = DataLoader(
            self.train_point_dataset,
            batch_size=self.config.TRAIN.batch_size,
            num_workers=self.config.TRAIN.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            collate_fn=MultiScannetppPointDataset.collate_fn,
        )

        self.train_iter = iter(self.train_loader)
        self.train_point_iter = iter(self.train_point_loader)

        self.val_dataset = MultiScannetppPointAABBDataset(
            self.config.DATASET.source_path,
            self.ply_path,
            self.config.DATASET.gt_ply_path,
            self.config.DATASET.val_split_path,
            self.config.DATASET.transform_path,
            crop_points=False if self.config.MODEL.input_type == "mesh" else True,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
        )

    def get_inner_dataset(self, scene_id, fixed=False):
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
                scaffold_decoder = DDP(scaffold_decoder, device_ids=[self.local_rank], find_unused_parameters=False)
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
                # if self.config.debug:
                #     ratio = valid_mask.sum() / valid_mask.numel()
                #     print(f"{idx}: valid: {valid_mask.sum()}/{valid_mask.numel()}, ratio: {ratio:.2f}")
                pred = pred[valid_mask]
                gt = gt[valid_mask]

            if self.config.MODEL.INIT.use_balance_weight:
                num_pos = gt.sum()
                num_neg = gt.numel() - num_pos
                pos_weight = num_neg / num_pos
                pos_weight = pos_weight.detach()
                # if self.config.debug:
                #     print(f"{idx}: shape: {gt.shape}, weight: {pos_weight:.2f}")
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
            assert scene_ids == point_batch["scene_id"], f"Scene ID mismatch: {scene_ids} vs {point_batch['scene_id']}"
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

            for inner_step_idx in range(self.num_inner_steps):

                loss_dict, metrics_dict, other_stuff = self.train_step(
                    batch, point_batch, inner_step_idx, step_idx,
                )
                loss = functools.reduce(torch.add, loss_dict.values())
                # loss_dict["total_loss"] = loss
                loss.backward()

                self.before_optimizer(step_idx)

                norm_dict = self.optimizer.get_max_norm()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Shouldn't encounter StopIteration here
                if inner_step_idx < self.num_inner_steps - 1:
                    batch = next(self.train_iter)

                loss_dict = {key: val.item() for key, val in loss_dict.items()}
                del loss

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

            if self.world_size > 1 and self.config.MODEL.decoder_type == "scaffold":
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

        # xyz_world = gs_params["xyz"]
        # rgb = gs_params["rgb"]
        # save_ply("xyz_world1.ply", xyz_world.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        # xyz_world = apply_transform(scaffold.xyz_voxel.float(), scaffold.transform)
        # save_ply("xyz_world2.ply", xyz_world.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        # features_3d = None
        # point_weights = torch.zeros(xyz_world.shape[0], device=xyz_world.device)

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
                            active_sh_degree=0,     # Does not matter since we are using RGB for color
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
                    grads[key] += (
                        grad_batch[j]
                        .nan_to_num(0)
                        .clamp(-self.config.MODEL.OPT.max_grad, self.config.MODEL.OPT.max_grad)
                        .detach()
                    )
                del grad_batch
                del loss

            for key in grads.keys():
                # normalize the channels by the largest values (inf-norm)
                denom = grads[key].norm(p=torch.inf, dim=0, keepdim=True).clamp_min(1e-12)
                grads[key] = grads[key] / denom

            grad_2d = grad_2d / torch.clamp_min(counter, 1)
            grad_2d_norm = grad_2d_norm / torch.clamp_min(counter, 1)

            grad_2d = grad_2d.view(-1, num_gs, 2)
            grad_2d = torch.mean(grad_2d, dim=1)

            grad_2d_norm = grad_2d_norm.view(-1, num_gs, 1)
            grad_2d_norm = torch.mean(grad_2d_norm, dim=1)
            return {
                "grad_input": grads,
                "grad_2d": grad_2d,
                "grad_2d_norm": grad_2d_norm,
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
    def evaluate_gs(
        self,
        scaffold: ScaffoldGSFull,
        scene_id: str,
        save_path: Optional[Path] = None,
        save_file_suffix: str = "",
        save_cat: bool = True,
        save_rgb_out: bool = True,
        save_depth_out: bool = True,
        use_identity: bool = False,
    ) -> List[Dict[str, float]]:
        """Compute the validation metrics"""
        dataset = ScannetppDataset(
            self.config.DATASET.source_path,
            self.ply_path,
            scene_id=scene_id,
            split="val",
            downsample=self.config.DATASET.image_downsample,
            load_depth=True,
            subsample_randomness=False,
        )

        val_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            # num_workers=0,
            num_workers=self.config.TRAIN.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        # Get the gs attributes
        params = scaffold.get_raw_params()
        if use_identity:
            gs_params = self.identity_decoder(
                latent=params["latent"],
                xyz_voxel=scaffold.xyz_voxel,
                transform=scaffold.transform,
                voxel_size=scaffold.voxel_size,
            )
        else:
            # Use the original decoder that was used for training
            gs_params = self.scaffold_decoder(
                latent=params["latent"],
                xyz_voxel=scaffold.xyz_voxel,
                transform=scaffold.transform,
                voxel_size=scaffold.voxel_size,
            )
        # Reshape the gs attributes from (N, M, C) to (N*M, C)
        for key in gs_params:
            gs_params[key] = gs_params[key].flatten(0, 1).contiguous()

        scale_norm = torch.mean(torch.norm(gs_params["scale"], p=2, dim=-1))
        opacity = torch.mean(gs_params["opacity"])

        # Compute the metrics
        metrics_list = []
        for batch_idx, batch_cpu in enumerate(val_loader):
            batch_gpu = self.move_to_device(batch_cpu)
            # batch_size is 1
            assert batch_gpu["rgb"].shape[0] == 1

            camera = Camera(
                fov_x=float(batch_cpu["fov_x"][0]),
                fov_y=float(batch_cpu["fov_y"][0]),
                height=int(batch_cpu["height"][0]),
                width=int(batch_cpu["width"][0]),
                image=batch_gpu["rgb"][0],
                world_view_transform=batch_gpu["world_view_transform"][0],
                full_proj_transform=batch_gpu["full_proj_transform"][0],
                camera_center=batch_gpu["camera_center"][0],
            )
            render_outputs = self.rasterizer(
                gs_params,
                camera,
                active_sh_degree=0,     # Does not matter
            )

            if self.config.MODEL.gaussian_type == "3d":
                # Additionally render the depth
                depth_expected = self.rasterizer.render_depth(gs_params, camera)
                render_outputs["depth_expected"] = depth_expected

            image_pred = torch.clamp(render_outputs["render"], 0, 1).unsqueeze(0)
            image_gt = torch.clamp(batch_gpu["rgb"], 0, 1)
            l1_loss = self.l1(image_pred, image_gt)
            psnr = self.psnr(image_pred, image_gt)
            ssim = self.ssim(image_pred, image_gt)
            lpips = self.lpips(image_pred, image_gt)

            complete_ratio = 0.0
            alpha_avg = 0.0
            if "alpha" in render_outputs:
                complete_ratio = torch.mean((render_outputs["alpha"] > 0.01).float()).item()
                alpha_avg = torch.mean(render_outputs["alpha"]).item()

            num_points = scaffold.num_points
            num_gs = gs_params["xyz"].shape[0]

            if save_path is not None:
                if save_cat:
                    image_combined = torch.cat([image_gt, image_pred], dim=-1)
                else:
                    image_combined = image_pred
                image_combined = image_combined[0].permute(1, 2, 0).cpu().numpy()
                image_combined = (image_combined * 255).astype("uint8")
                image_combined = Image.fromarray(image_combined)
                image_combined = image_combined.resize((image_combined.width, image_combined.height))
                save_path.mkdir(parents=True, exist_ok=True)
                if save_file_suffix == "":
                    if save_rgb_out:
                        image_combined.save(save_path / f"{scene_id}_{batch_idx:04d}.jpg")
                    save_depth_path = save_path / f"{scene_id}_{batch_idx:04d}_depth.jpg"
                else:
                    if save_rgb_out:
                        image_combined.save(save_path / f"{scene_id}_{batch_idx:04d}_{save_file_suffix}.jpg")
                    save_depth_path = save_path / f"{scene_id}_{batch_idx:04d}_{save_file_suffix}_depth.jpg"

                if "depth_expected" in render_outputs:
                    if save_cat:
                        depth_combined = torch.cat([batch_gpu["depth"], render_outputs["depth_expected"]], dim=-1)
                    else:
                        depth_combined = render_outputs["depth_expected"]
                    if save_depth_out:
                        save_depth_opencv(depth_combined, save_depth_path)

            metrics_dict = {
                "l1_loss": l1_loss.item(),
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "scale": scale_norm.item(),
                "opacity": opacity.item(),
                "num_points": num_points,
                "num_gs": num_gs,
                "complete_ratio": complete_ratio,
                "alpha_avg": alpha_avg,
            }

            if "depth_expected" in render_outputs:
                expected_depth_metrics = compute_full_depth_metrics(render_outputs["depth_expected"], batch_gpu["depth"])
                metrics_dict.update({f"depth_expected_{k}": v.item() for k, v in expected_depth_metrics.items()})
            if "depth_median" in render_outputs:
                median_depth_metrics = compute_full_depth_metrics(render_outputs["depth_median"], batch_gpu["depth"])
                metrics_dict.update({f"depth_median_{k}": v.item() for k, v in median_depth_metrics.items()})
            metrics_list.append(metrics_dict)
        return metrics_list

    def load_scene_point_pair(
        self,
        scene_id: str,
    ):
        xyz, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel = self.val_dataset.load_voxelized_colmap_points(
            scene_id,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
            crop_points=True,
        )
        xyz_gt, rgb_gt, xyz_voxel_gt, *_, normal_gt = self.val_dataset.load_voxelized_mesh_points(
            scene_id,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
            bbox_min=bbox[0],
            bbox_max=bbox[1],
            load_normal=True,
        )

        xyz = torch.from_numpy(xyz).float().to(self.device)
        rgb = torch.from_numpy(rgb).float().to(self.device)
        xyz_voxel = torch.from_numpy(xyz_voxel).float().to(self.device)
        voxel_to_world = torch.from_numpy(world_to_voxel).float().to(self.device).inverse()
        bbox_voxel = torch.from_numpy(bbox_voxel).int().to(self.device)

        xyz_gt = torch.from_numpy(xyz_gt).float().to(self.device)
        rgb_gt = torch.from_numpy(rgb_gt).float().to(self.device)
        xyz_voxel_gt = torch.from_numpy(xyz_voxel_gt).int().to(self.device)
        normal_gt = torch.from_numpy(normal_gt).float().to(self.device)
        return {
            "xyz_voxel": xyz_voxel,
            "rgb": rgb,
            "bbox_voxel": bbox_voxel,
            "xyz_voxel_gt": xyz_voxel_gt,
            "normal_gt": normal_gt,
            "voxel_to_world": voxel_to_world,
            "xyz": xyz,
            "xyz_gt": xyz_gt,
            "rgb_gt": rgb_gt,
        }

    def after_train_step(self, step_idx: int, force_reset: bool = False):
        reset_dataloader = False
        if step_idx % self.config.MODEL.OPT.add_step_every == 0:
            if self.num_inner_steps < self.config.MODEL.OPT.num_steps:
                self.num_inner_steps += 1
                CONSOLE.print(f"Adding step, num_inner_steps: {self.num_inner_steps}")

                reset_dataloader = True

        if reset_dataloader or force_reset:

            self.data_generator = np.random.default_rng(self.config.MODEL.SCAFFOLD.seed + self.local_rank + step_idx)
            self.data_point_generator = np.random.default_rng(self.config.MODEL.SCAFFOLD.seed + self.local_rank + step_idx)

            sampler = RepeatSampler(
                self.train_dataset,
                batch_size=self.config.TRAIN.batch_size,
                num_repeat=self.num_inner_steps,
                generator=self.data_generator,
            )

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.TRAIN.batch_size,
                num_workers=self.config.TRAIN.num_workers,
                # num_workers=0,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
            )
            self.train_iter = iter(self.train_loader)

            sampler = RepeatSampler(
                self.train_point_dataset,
                batch_size=self.config.TRAIN.batch_size,
                num_repeat=1,
                generator=self.data_point_generator,
            )

            self.train_point_loader = DataLoader(
                self.train_point_dataset,
                batch_size=self.config.TRAIN.batch_size,
                num_workers=self.config.TRAIN.num_workers,
                # num_workers=0,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
                collate_fn=MultiScannetppPointDataset.collate_fn,
            )
            self.train_point_iter = iter(self.train_point_loader)

        # Empty cuda cache
        torch.cuda.empty_cache()
