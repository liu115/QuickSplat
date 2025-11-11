from typing import Any, Union, Optional, List, Tuple, Dict, Literal
from pathlib import Path
import json
import functools

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from dataset.scannetpp import ScannetppDataset, MultiScannetppPointDataset
from dataset.cache_loader import GPUCacheLoader
from trainers.phase2_trainer import Phase2Trainer
from models.scaffold_gs import ScaffoldGSFull
from modules.rasterizer_2d import Full2DGSRasterizer
from modules.gaussian_2d import Gaussian2DModel
from modules.rasterizer_3d import Camera

from utils.depth import compute_full_depth_metrics, save_depth_opencv
from utils.rich_utils import CONSOLE
from utils.metrics import Timer
from utils.sparse import xyz_list_to_bxyz, chamfer_dist_with_crop, chamfer_dist_mesh_with_crop
from utils.fusion import MeshExtractor


# This voxel size is used for volumetric fusion during evaluation. Don't have to be the same as training voxel size.
EVAL_VOXEL_SIZE = 0.02
MAX_STEPS_TRAIN = 5
EVAL_MESH_POINTS = 200_000


class InferenceTrainer(Phase2Trainer):

    def setup_optimizers(self):
        # No need to setup optimizers for inference
        pass

    def setup_dataloader(self):
        self.test_dataset = MultiScannetppPointDataset(
            self.config.DATASET.source_path,
            self.config.DATASET.ply_path,
            self.config.DATASET.gt_ply_path,
            # Use test_split for inference and evaluation
            self.config.DATASET.test_split_path,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
        )

        # Make val_dataset the same as test_dataset so that that functions in Phase2Trainer would work
        self.val_dataset = self.test_dataset

    def setup_modules(self):
        super().setup_modules()
        self.mesh_extractor = MeshExtractor(
            gaussian_type=self.config.MODEL.gaussian_type,
            voxel_size=EVAL_VOXEL_SIZE,
            sdf_trunc=EVAL_VOXEL_SIZE * 4,
        )

    def load_checkpoint(self, ckpt_path: str) -> None:
        if Path(ckpt_path).is_dir():
            ckpt_files = list(Path(ckpt_path).glob("*.ckpt"))
            assert len(ckpt_files) > 0, f"No checkpoint found in {ckpt_path}"
            ckpt_files.sort()
            ckpt_path = str(ckpt_files[-1])
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model.load_state_dict(ckpt["model"])

        if self.config.MODEL.decoder_type == "scaffold":
            self.scaffold_decoder.load_state_dict(ckpt["decoder"])
        else:
            # identity decoder does not have parameters
            pass

        if self.config.MODEL.DENSIFIER.enable:
            self.densifier.load_state_dict(ckpt["densifier"])

        # Load initializer here
        if "initializer" in ckpt:
            self.initializer.load_state_dict(ckpt["initializer"])

    @torch.no_grad()
    def init_scaffold_test(self, scene_id: str, test_mode: bool = False) -> ScaffoldGSFull:
        assert test_mode, "Test mode must be True in init_scaffold_test"
        # crop_points = True
        xyz, rgb, xyz_voxel, xyz_offset, bbox, bbox_voxel, world_to_voxel = self.val_dataset.load_voxelized_colmap_points(
            scene_id,
            voxel_size=self.config.MODEL.SCAFFOLD.voxel_size,
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

    def _single_iteration_with_densify(
        self,
        inner_step_idx: int,
        scene_id: str,
        scaffold: ScaffoldGSFull,
    ):
        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps, max_steps_train=MAX_STEPS_TRAIN)
        CONSOLE.print(f"inner={inner_step_idx}: timestamp={timestamp}")

        params = scaffold.get_raw_params()
        grad_dict = self.get_params_grad(scaffold, params, scene_id, fixed=True)
        grads = grad_dict["grad_input"]
        params = {k: v.detach() for k, v in params.items()}
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
                max_gt_samples_training=sample_n_last,
                # max_gt_samples_training=self.config.DATASET.eval_max_num_points,
                max_samples_second_last=samples_second_last,
            )
            output_sparse = dense_outputs["out"]
            ignore_final = dense_outputs["ignore_final"]

            # Select only those that exists
            before_prune = output_sparse
            output_sparse = self.prune(output_sparse, ~ignore_final)
            CONSOLE.print(f"inner={inner_step_idx}: generating new points: {output_sparse.shape[0]} / {before_prune.shape[0]}")

            xyz_new, latent_new = output_sparse.decomposed_coordinates_and_features
            xyz_new = xyz_new[0]
            latent_new = latent_new[0]
            grad_new = torch.zeros_like(latent_new)

            xyz_new = torch.cat([scaffold.xyz_voxel, xyz_new], dim=0)
            latent_new = torch.cat([params["latent"], latent_new], dim=0)
            grad_new = torch.cat([grads["latent"], grad_new], dim=0)

        if self.config.MODEL.DENSIFIER.recompute_grad:
            latent_new = latent_new.clone().detach().requires_grad_(True)
            CONSOLE.print("Recompute_grad")
            scaffold.set_raw_params_and_xyz({"latent": latent_new}, xyz_new)
            grad_dict = self.get_params_grad(scaffold, {"latent": latent_new}, scene_id, fixed=True)
            grad_new = grad_dict["grad_input"]["latent"]
        else:
            # Use zero gradients for new points
            pass

        with torch.no_grad():
            bxyz, _ = xyz_list_to_bxyz([xyz_new])
            outputs = self.model(
                bxyz,
                {"latent": latent_new},
                {"latent": grad_new},
                timestamp=timestamp,
            )

            latent_delta = (
                outputs["latent"]
                * self.config.MODEL.OPT.output_scale
                * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
            )

        latent_new = latent_new + latent_delta
        scaffold.set_raw_params_and_xyz({"latent": latent_new}, xyz_new)
        return scaffold

    def _single_iteration(self, inner_step_idx: int, scene_id: str, scaffold: ScaffoldGSFull):
        timestamp = self._get_timestamp(inner_step_idx, self.num_inner_steps, max_steps_train=MAX_STEPS_TRAIN)
        CONSOLE.print(f"inner={inner_step_idx}: timestamp={timestamp}")
        params = scaffold.get_raw_params()

        grad_dict = self.get_params_grad(scaffold, params, scene_id, fixed=True)
        grads = grad_dict["grad_input"]
        params = {k: v.detach() for k, v in params.items()}

        with torch.no_grad():
            bxyz, _ = xyz_list_to_bxyz([scaffold.xyz_voxel])
            outputs = self._model(bxyz, params, grads, timestamp=timestamp)

            latent_delta = (
                outputs["latent"]
                * self.config.MODEL.OPT.output_scale
                * max(self.config.MODEL.OPT.scale_gamma ** inner_step_idx, self.config.MODEL.OPT.min_scale)
            )
        latent_new = params["latent"] + latent_delta

        scaffold.set_raw_params({"latent": latent_new})
        return scaffold

    def train(self):
        self.model.eval()
        self.scaffold_decoder.eval()
        if self.config.MODEL.DENSIFIER.enable:
            self.densifier.eval()

        self.all_timer = Timer()
        scene_list = self.test_dataset.scene_list
        self.num_inner_steps = self.config.MODEL.OPT.num_steps

        all_scene_history = {}
        for scene_id in scene_list:
            torch.cuda.reset_peak_memory_stats()
            scene_history = []

            save_dir = self.output_dir / "outputs"
            self.all_timer.start()
            scaffold = self.init_scaffold_test(scene_id, test_mode=True)

            init_metrics: List[Dict[str, float]]
            mem_usage = torch.cuda.max_memory_allocated()

            with self.all_timer.pause_context():
                # Pause during evaluation
                init_metrics = self.evaluate_gs(
                    scaffold=scaffold,
                    scene_id=scene_id,
                    save_path=None,
                    save_file_suffix="init",
                    use_identity=False,
                )
                # Average over all the frames
                init_metrics = {x: np.mean([m[x] for m in init_metrics]) for x in init_metrics[0].keys()}
                mesh_metrics = self.evaluate_mesh_and_save_scaffold(
                    scene_id,
                    scaffold,
                    # Uncomment to save intermediate meshes
                    # save_path=save_dir / f"{scene_id}_0.ply",
                    depth_source="expected",
                )
                init_metrics.update(mesh_metrics)
                scene_history.append({
                    "scene_id": scene_id,
                    "step": 0,
                    "time": self.all_timer.current(),
                    "metrics": init_metrics,
                    "mem_usage": mem_usage,
                    "is_finetune": False,
                })

            # Run the Quicksplat optimization
            for inner_step_idx in range(self.num_inner_steps):

                if self.config.MODEL.DENSIFIER.enable and inner_step_idx < self.config.MODEL.DENSIFIER.num_densify_steps_eval:
                    scaffold = self._single_iteration_with_densify(inner_step_idx, scene_id, scaffold)
                else:
                    scaffold = self._single_iteration(inner_step_idx, scene_id, scaffold)
                mem_usage = torch.cuda.max_memory_allocated()

                with self.all_timer.pause_context():
                    # Pause during evaluation
                    eval_metrics = self.evaluate_gs(
                        scaffold=scaffold,
                        scene_id=scene_id,
                        save_path=save_dir / scene_id,
                    )
                    # Average over all the frames
                    eval_metrics = {x: np.mean([m[x] for m in eval_metrics]) for x in eval_metrics[0].keys()}
                    record = {
                        "scene_id": scene_id,
                        "step": inner_step_idx + 1,
                        "metrics": eval_metrics,
                        # "lr": lr,
                        "mem_usage": mem_usage,
                        "time": self.all_timer.current(),
                    }

                    mesh_metrics = self.evaluate_mesh_and_save_scaffold(
                        scene_id,
                        scaffold,
                        # Uncomment to save intermediate meshes
                        # save_path=save_dir / f"{scene_id}_{inner_step_idx + 1}.ply",
                        depth_source="expected",
                    )
                    eval_metrics.update(mesh_metrics)

                    scene_history.append({
                        "scene_id": scene_id,
                        "step": inner_step_idx + 1,
                        "time": self.all_timer.current(),
                        "metrics": eval_metrics,
                        "mem_usage": mem_usage,
                        "is_finetune": False,
                    })

                    record.update(mesh_metrics)
                    print(json.dumps(record, indent=4))

            torch.cuda.empty_cache()

            # Per-scene optimization using 2DGS
            finetune_history = self.finetune(scene_id, scaffold)
            scene_history.extend(finetune_history)

            # Visualize the metrics
            psnr_list = [m["metrics"]["psnr"] for m in scene_history]
            chamfer_list = [m["metrics"]["chamfer_bi"] for m in scene_history]
            time_list = [m["time"] for m in scene_history]
            depth_loss_list = [m["metrics"]["depth_expected_l1"] for m in scene_history]
            plt.clf()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].plot(time_list, psnr_list)
            axs[0].set_xlabel("Time")
            axs[0].set_ylabel("PSNR")
            axs[1].plot(time_list, chamfer_list)
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Chamfer Distance")
            axs[2].plot(time_list, depth_loss_list)
            axs[2].set_xlabel("Time")
            axs[2].set_ylabel("Depth (expected)")
            plt.savefig(self.output_dir / f"{scene_id}_test_metrics.png")
            all_scene_history[scene_id] = scene_history

            self.all_timer.reset()
            # End of per-scene loop

        # Print the average metrics across scenes
        output_summary = {
            "per_scene": {
                scene_id: {} for scene_id in all_scene_history.keys()
            },
            "average": {},
        }
        metric_names = [
            "psnr",
            "lpips",
            "chamfer_bi",
            "chamfer_bi2",
            "depth_expected_l1",
            "depth_expected_log_l1",
            "depth_expected_accuracy_0.02",
            "depth_expected_accuracy_0.05",
            "depth_expected_accuracy_0.1",
            "depth_expected_accuracy_0.2",
            "depth_expected_delta1",
        ]
        for metric_name in metric_names:
            avg_val_init = 0.0
            avg_val_opt = 0.0
            avg_val_ft = 0.0
            for scene_id, scene_history in all_scene_history.items():
                val_init = scene_history[0]["metrics"][metric_name]
                val_opt = scene_history[self.num_inner_steps]["metrics"][metric_name]
                val_ft = scene_history[-1]["metrics"][metric_name]
                output_summary["per_scene"][scene_id][metric_name] = {
                    "init": val_init,
                    "opt": val_opt,
                    "ft": val_ft,
                }
                avg_val_init += val_init
                avg_val_opt += val_opt
                avg_val_ft += val_ft
            avg_val_ft /= len(all_scene_history)
            avg_val_init /= len(all_scene_history)
            avg_val_opt /= len(all_scene_history)
            CONSOLE.print(f"Metric: {metric_name}, Init: {avg_val_init:.3f}, Opt: {avg_val_opt:.3f}, FT: {avg_val_ft:.3f}")

            output_summary["average"][metric_name] = {
                "init": avg_val_init,
                "opt": avg_val_opt,
                "ft": avg_val_ft,
            }
        with open(self.output_dir / "test_summary.json", "w") as f:
            json.dump(output_summary, f, indent=4)

    def evaluate_mesh_and_save_scaffold(
        self,
        scene_id: str,
        scaffold: ScaffoldGSFull,
        save_path: Optional[Path] = None,
        depth_source: Literal["median", "expected"] = "expected",
        depth_trunc: float = 8.0,
    ) -> None:
        params = scaffold.get_raw_params()
        gaussian_params = self.scaffold_decoder(
            latent=params["latent"],
            xyz_voxel=scaffold.xyz_voxel,
            transform=scaffold.transform,
            voxel_size=scaffold.voxel_size,
        )
        # Reshape the gs attributes from (N, M, C) to (N*M, C)
        for key in gaussian_params:
            gaussian_params[key] = gaussian_params[key].flatten(0, 1).contiguous()

        return self.evaluate_mesh_and_save(
            scene_id,
            gaussian_params,
            save_path,
            depth_source=depth_source,
            depth_trunc=depth_trunc,
        )

    def evaluate_mesh_and_save(
        self,
        scene_id,
        gaussian_params,
        save_path: Optional[Path] = None,
        depth_source: Literal["median", "expected"] = "expected",
        depth_trunc: float = 8.0,
    ):
        # xyz_gt, rgb = self.val_dataset.load_mesh_points(scene_id)
        dataset = ScannetppDataset(
            self.config.DATASET.source_path,
            self.config.DATASET.gt_ply_path,
            scene_id=scene_id,
            split="train",
            downsample=self.config.DATASET.image_downsample,
            # num_train_frames=self.config.DATASET.num_train_frames,
            # subsample_randomness=False,
        )
        xyz_gt, rgb = dataset.load_points()
        verts, faces = self.mesh_extractor.reconstruct_and_save(
            dataset,
            gaussian_params,
            save_path,
            depth_source=depth_source,
            depth_trunc=depth_trunc,
        )
        xyz_gt = torch.from_numpy(xyz_gt).float().to(self.device)
        verts = verts.float().to(self.device)

        # Replace this with point sampling from triangle mesh surfaces
        # chamfer_bi = chamfer_dist_with_crop(verts, xyz_gt, norm=1, direction="bidirection")
        # chamfer_bi2 = chamfer_dist_with_crop(verts, xyz_gt, norm=2, direction="bidirection")
        # chamfer_gt_to_pred = chamfer_dist_with_crop(verts, xyz_gt, norm=1, direction="gt_to_pred")
        # chamfer_pred_to_gt = chamfer_dist_with_crop(verts, xyz_gt, norm=1, direction="pred_to_gt")

        verts_gt, verts_rgb_gt, faces_gt = dataset.load_mesh()
        verts_gt = torch.from_numpy(verts_gt).float().to(self.device)
        faces_gt = torch.from_numpy(faces_gt).float().to(self.device)
        chamfer_bi_mesh = chamfer_dist_mesh_with_crop(
            verts, faces,
            verts_gt, faces_gt,
            norm=1,
            direction="bidirection",
            num_sampled_points=EVAL_MESH_POINTS,
        )
        chamfer_bi2_mesh = chamfer_dist_mesh_with_crop(
            verts, faces,
            verts_gt, faces_gt,
            norm=2,
            direction="bidirection",
            num_sampled_points=EVAL_MESH_POINTS,
        )
        return {
            "num_vertices": verts.shape[0],
            "num_faces": faces.shape[0],
            "num_gt_points": xyz_gt.shape[0],
            "chamfer_bi": chamfer_bi_mesh.item(),
            "chamfer_bi2": chamfer_bi2_mesh.item(),
        }

    def finetune(self, scene_id, scaffold):
        assert self.config.MODEL.gaussian_type == "2d", "Only 2DGS finetuning is supported"
        finetune_history = []

        # Create the dataset
        train_dataset = ScannetppDataset(
            self.config.DATASET.source_path,
            self.config.DATASET.ply_path,
            scene_id=scene_id,
            split="train",
            downsample=self.config.DATASET.image_downsample,
        )
        if self.config.DATASET.cache_gpu:
            # Store all the data in GPU memory for faster access
            train_loader = GPUCacheLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                drop_last=False,
                device=self.device,
                verbose=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=self.config.TRAIN.num_workers,
                pin_memory=False,
                drop_last=False,
            )
        train_iter = iter(train_loader)

        val_dataset = ScannetppDataset(
            self.config.DATASET.source_path,
            self.config.DATASET.ply_path,
            scene_id=scene_id,
            split="val",
            # white_background=True,
            downsample=self.config.DATASET.image_downsample,
            subsample_randomness=False,
            load_depth=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        # Initialize the model
        params = scaffold.get_raw_params()
        gs_params = self.scaffold_decoder(
            latent=params["latent"],
            xyz_voxel=scaffold.xyz_voxel,
            transform=scaffold.transform,
            voxel_size=scaffold.voxel_size,
        )
        # Reshape the gs attributes from (N, M, C) to (N*M, C)
        for key in gs_params:
            gs_params[key] = gs_params[key].flatten(0, 1).contiguous()

        if self.config.MODEL.OPT.prune_opacity_ft:
            valid_mask = gs_params["opacity"].squeeze(-1) > 0.005
            print(f"Pruning {gs_params['opacity'].shape[0] - valid_mask.sum()}/{gs_params['opacity'].shape[0]} points")
            gs_params = {k: v[valid_mask].contiguous() for k, v in gs_params.items()}

        if self.config.MODEL.OPT.reset_opacity_ft:
            gs_params["opacity"] = torch.ones_like(gs_params["opacity"]) * 0.01

        num_iterations = self.config.TRAIN.num_iterations
        gaussian_model = Gaussian2DModel.create_from_gaussians(
            self.config.MODEL.GSPLAT,
            gs_params["xyz"],
            gs_params["rgb"],
            gs_params["scale"],
            gs_params["rotation"],
            gs_params["opacity"],
        )
        gaussian_model.training_setup(self.config.MODEL.GSPLAT)

        self.bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)   # White
        rasterizer = Full2DGSRasterizer(
            self.config.MODEL.GSPLAT,
            self.bg_color,
        )

        for step_idx in tqdm(range(0, num_iterations)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            output_dict, loss_dict, metrics_dict = self.finetune_step(batch, gaussian_model, step_idx, rasterizer)
            loss_dict["total_loss"].backward()

            # self.model.update_learning_rate(step_idx)
            # if step_idx % 1000 == 0:
            #     self.model.upate_sh_degree()

            if step_idx % self.config.TRAIN.val_interval == 0:
                with self.all_timer.pause_context():
                    # Do evaluation
                    val_metrics = self.finetune_eval(gaussian_model, val_loader, rasterizer, scene_id, step_idx)
                    finetune_history.append({
                        "step": step_idx,
                        "time": self.all_timer.current(),
                        "metrics": val_metrics,
                        "is_finetune": True,
                    })
                    print(json.dumps(val_metrics, indent=4))

            # NOTE: No densification during finetuning
            # with torch.no_grad():
            #     if self.config.MODEL.GSPLAT.density_control:
            #         if step_idx < self.config.MODEL.GSPLAT.densify_until_iter:
            #             # Keep track of max radii in image-space for pruning
            #             visibility_filter = output_dict["visibility_filter"]
            #             radii = output_dict["radii"]
            #             viewspace_points = output_dict["viewspace_points"]
            #             gaussian_model.max_radii2D[visibility_filter] = torch.max(
            #                 gaussian_model.max_radii2D[visibility_filter],
            #                 radii[visibility_filter],
            #             )
            #             gaussian_model.add_densification_stats(
            #                 viewspace_points,
            #                 visibility_filter,
            #             )

            #             if (
            #                 step_idx > self.config.MODEL.GSPLAT.densify_from_iter
            #                 and step_idx % self.config.MODEL.GSPLAT.densification_interval == 0
            #             ):
            #                 size_threshold = 20 if step_idx > self.config.MODEL.GSPLAT.opacity_reset_interval else None
            #                 gaussian_model.densify_and_prune(
            #                     self.config.MODEL.GSPLAT.densify_grad_threshold,
            #                     0.005,
            #                     self.train_dataset.scene_scale,
            #                     size_threshold,
            #                 )
            # Step the optimizers
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)

        with self.all_timer.pause_context():
            # Evaluate the mesh
            val_metrics = self.finetune_eval(gaussian_model, val_loader, rasterizer, scene_id, num_iterations)
            gaussian_params = {
                "xyz": gaussian_model.get_xyz,
                "rgb": gaussian_model.get_rgb,
                "scale": gaussian_model.get_scaling,
                "rotation": gaussian_model.get_rotation,
                "opacity": gaussian_model.get_opacity,
            }

        # Don't pause timer at the end to include mesh extraction time
        self.evaluate_mesh_and_save(
            scene_id,
            gaussian_params,
            save_path=self.output_dir / "outputs" / f"{scene_id}_ft.ply",
            depth_source="expected",
        )
        finetune_history.append({
            "step": num_iterations,
            "time": self.all_timer.current(),
            "metrics": val_metrics,
            "is_finetune": True,
        })
        return finetune_history

    def finetune_step(self, batch, model, step_idx, rasterizer):
        batch = self.move_to_device(batch)
        # orig_batch = batch
        loss_dict = {}
        metrics_dict = {}
        camera = Camera(
            fov_x=float(batch["fov_x"][0].item()),
            fov_y=float(batch["fov_y"][0].item()),
            height=int(batch["height"][0].item()),
            width=int(batch["width"][0].item()),
            image=batch["rgb"][0],
            world_view_transform=batch["world_view_transform"][0],
            full_proj_transform=batch["full_proj_transform"][0],
            camera_center=batch["camera_center"][0],
        )
        outputs = rasterizer(model, camera)
        # Compute losses
        l1_loss = self.l1(outputs["render"], batch["rgb"][0])
        loss_dict["l1_loss"] = l1_loss

        ssim_loss = 1 - self.ssim(outputs["render"].unsqueeze(0), batch["rgb"])
        if self.config.MODEL.GSPLAT.lambda_dssim > 0:
            loss_dict["ssim_loss"] = ssim_loss * self.config.MODEL.GSPLAT.lambda_dssim

        if self.config.MODEL.GSPLAT.lambda_dist > 0 and step_idx > self.config.MODEL.GSPLAT.dist_start_iter:
            # Compute distortion loss
            distortion = outputs["distortion"]
            loss_dict["dist_loss"] = distortion.mean() * self.config.MODEL.GSPLAT.lambda_dist

        if self.config.MODEL.GSPLAT.lambda_normal > 0 and step_idx > self.config.MODEL.GSPLAT.normal_start_iter:
            # Compute normal loss
            normal_from_depth = outputs["normal_from_depth"]
            normal = outputs["normal"]
            normal_error = (1 - (normal * normal_from_depth).sum(dim=0))
            loss_dict["normal_loss"] = normal_error.mean() * self.config.MODEL.GSPLAT.lambda_normal

        loss = functools.reduce(torch.add, loss_dict.values())
        # loss.backward()
        loss_dict["total_loss"] = loss

        with torch.no_grad():
            metrics_dict["psnr"] = self.psnr(
                torch.clamp(outputs["render"], 0, 1),
                torch.clamp(batch["rgb"][0], 0, 1)
            )
            metrics_dict["ssim"] = self.ssim(
                torch.clamp(outputs["render"], 0, 1).unsqueeze(0),
                torch.clamp(batch["rgb"], 0, 1)
            )
            metrics_dict["num_points"] = model.get_xyz.shape[0]
            # metrics_dict["lpips"] = self.lpips(outputs["rgb"].unsqueeze(0), batch["rgb"])

        # Additionally maintain the visibility and the gradients
        output_dict = {
            "visibility_filter": outputs["visibility_filter"],
            "radii": outputs["radii"],
            "viewspace_points": outputs["viewspace_points"],
        }
        return output_dict, loss_dict, metrics_dict

    @torch.no_grad()
    def finetune_eval(
        self,
        model,
        val_loader,
        rasterizer,
        scene_id,
        step_idx: int,
        save_outputs: bool = False,
    ) -> Dict[str, float]:
        metrics_list = []
        opacity = torch.mean(model.get_opacity).item()
        scale_norm = torch.mean(torch.norm(model.get_scaling, p=2, dim=-1)).item()

        for batch_idx, batch in enumerate(val_loader):
            orig_batch = batch
            batch = self.move_to_device(batch)
            camera = Camera(
                fov_x=float(orig_batch["fov_x"][0]),
                fov_y=float(orig_batch["fov_y"][0]),
                height=int(orig_batch["height"][0]),
                width=int(orig_batch["width"][0]),
                image=batch["rgb"][0],
                world_view_transform=batch["world_view_transform"][0],
                full_proj_transform=batch["full_proj_transform"][0],
                camera_center=batch["camera_center"][0],
            )
            outputs = rasterizer(model, camera)
            complete_ratio = 0.0
            alpha_avg = 0.0
            if "alpha" in outputs:
                complete_ratio = torch.mean((outputs["alpha"] > 0.01).float()).item()
                alpha_avg = torch.mean(outputs["alpha"]).item()

            image_pred = torch.clamp(outputs["render"], 0, 1).unsqueeze(0)
            image_gt = torch.clamp(batch["rgb"], 0, 1)
            l1_loss = self.l1(image_pred, image_gt)
            psnr = self.psnr(image_pred, image_gt)
            ssim = self.ssim(image_pred, image_gt)
            lpips = self.lpips(image_pred, image_gt)

            num_points = model.get_xyz.shape[0]
            metrics_dict = {
                "l1_loss": l1_loss.item(),
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                # "depth_expected_loss": depth_expected_loss.item(),
                # "depth_median_loss": depth_median_loss.item(),
                "num_points": num_points,
                "opacity": opacity,
                "scale": scale_norm,
                "complete_ratio": complete_ratio,
                "alpha_avg": alpha_avg,
            }
            expected_depth_metrics = compute_full_depth_metrics(outputs["depth_expected"], batch["depth"])
            median_depth_metrics = compute_full_depth_metrics(outputs["depth_median"], batch["depth"])
            metrics_dict.update({f"depth_expected_{k}": v.item() for k, v in expected_depth_metrics.items()})
            metrics_dict.update({f"depth_median_{k}": v.item() for k, v in median_depth_metrics.items()})
            metrics_list.append(metrics_dict)

            if save_outputs:
                save_path = self.output_dir / "outputs" / scene_id
                save_path.mkdir(parents=True, exist_ok=True)
                save_rgb_path = save_path / f"{scene_id}_{batch_idx:04d}_ft.jpg"
                image_combined = torch.cat([image_gt, image_pred], dim=-1)
                image_combined = image_combined[0].permute(1, 2, 0).cpu().numpy()
                image_combined = (image_combined * 255).astype("uint8")
                image_combined = Image.fromarray(image_combined)
                image_combined.save(save_rgb_path)

                save_depth_path = save_path / f"{scene_id}_{batch_idx:04d}_ft_depth.jpg"
                depth_combined = torch.cat([batch["depth"], outputs["depth_expected"]], dim=-1)
                save_depth_opencv(depth_combined, save_depth_path)

        metrics_dict = {}
        for k, v in metrics_list[0].items():
            metrics_dict[k] = np.mean([m[k] for m in metrics_list])

        gaussian_params = {
            "xyz": model.get_xyz,
            "rgb": model.get_rgb,
            "scale": model.get_scaling,
            "rotation": model.get_rotation,
            "opacity": model.get_opacity,
        }
        mesh_metrics = self.evaluate_mesh_and_save(
            scene_id,
            gaussian_params,
            save_path=None,
            # Save the intermediate mesh for visualization and debugging
            depth_source="expected",
        )
        metrics_dict.update(mesh_metrics)

        return metrics_dict
