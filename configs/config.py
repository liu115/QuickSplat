from yacs.config import CfgNode as CN


_C = CN()
_C.trainer = "gaussian"
_C.name = "default"
_C.subname = "default"
_C.params = ""
_C.log_dir = "/tmp/log"
_C.save_dir = "/tmp/saved"
_C.save_only_latest_checkpoint = True
_C.save_per_scene = False
_C.save_checkpoint = True
_C.debug = False
_C.eval_mode = False
_C.do_logging = True

_C.MACHINE = CN()
_C.MACHINE.seed = 42
_C.MACHINE.num_devices = 1
_C.MACHINE.num_machines = 1
_C.MACHINE.machine_rank = 0
_C.MACHINE.device_type = "cuda"
_C.MACHINE.dist_url = "auto"


_C.DATASET = CN()
_C.DATASET.source_path = "SOURCE_PATH"
_C.DATASET.ply_path = "PLY_PATH"
_C.DATASET.gt_ply_path = "GT_PLY_PATH"
_C.DATASET.num_train_frames = -1        # Use all training frames
_C.DATASET.train_split_path = "TRAIN_SPLIT_PATH"
_C.DATASET.val_split_path = "VAL_SPLIT_PATH"
_C.DATASET.test_split_path = "TEST_SPLIT_PATH"
_C.DATASET.train_chunk_path = "TRAIN_CHUNK_PATH"
_C.DATASET.val_chunk_path = "VAL_CHUNK_PATH"
_C.DATASET.transform_path = "TRANSFORM_PATH"
_C.DATASET.image_downsample = 2
_C.DATASET.cache_gpu = False        # Speed up the training (for GS methods)

_C.DATASET.overide_scene_id = None
_C.DATASET.num_views = 5
_C.DATASET.num_neighbor_views = 5
_C.DATASET.num_sample_from_neighbors = 8
_C.DATASET.num_random_views = 0
_C.DATASET.num_support_views = 0
_C.DATASET.num_target_views = 1
_C.DATASET.num_inner_target_views = 0
_C.DATASET.num_gs_per_voxel = 1
_C.DATASET.use_point_aug = False
_C.DATASET.use_point_rotate_aug = False
_C.DATASET.use_point_color_aug = False
_C.DATASET.voxelize_aug = False
_C.DATASET.max_num_points = 500_000
_C.DATASET.eval_max_num_points = 1_000_000
_C.DATASET.chunk_size = 0.0     # > 0.0 to enable chunking for SGNN_FULL Trainer


_C.TRAIN = CN()
_C.TRAIN.batch_size = 4
_C.TRAIN.num_workers = 4
_C.TRAIN.num_iterations = 1000
_C.TRAIN.num_epochs = 0
_C.TRAIN.grad_acc = 1
_C.TRAIN.log_interval = 50
_C.TRAIN.val_interval = 100
_C.TRAIN.save_interval = 1000000
_C.TRAIN.ckpt_path = ""
# _C.TRAIN.initializer_path = ""


_C.MODEL = CN()

# New trainer
# "2d", "3d"
_C.MODEL.gaussian_type = "2d"
# "scaffold", "identity"
_C.MODEL.decoder_type = "scaffold"
# neural, with_memory, with_feature
_C.MODEL.model_type = "neural"
# mesh, colmap, colmap+completion
_C.MODEL.input_type = "mesh"
# xyz+rgb, xyz+rgb+normal, xyz+rgb+normal+scale
# _C.MODEL.init_type = "xyz+rgb"
_C.MODEL.use_identity_loss = False

# "dino", "dino_decoder"
_C.MODEL.image_encoder_type = "dino"

_C.MODEL.scale_2d_reg_mult = 0.0
_C.MODEL.scale_3d_reg_mult = 0.0
_C.MODEL.opacity_reg_mult = 0.0

_C.MODEL.SCAFFOLD = CN()
_C.MODEL.SCAFFOLD.voxel_size = 0.02
_C.MODEL.SCAFFOLD.hidden_dim = 128
_C.MODEL.SCAFFOLD.num_layers = 2
_C.MODEL.SCAFFOLD.num_gs = 1
_C.MODEL.SCAFFOLD.skip_connect = False
# Seed to initialize the latent vector
_C.MODEL.SCAFFOLD.seed = 5566
_C.MODEL.SCAFFOLD.fix_init = True
_C.MODEL.SCAFFOLD.max_scale = 10.0
_C.MODEL.SCAFFOLD.quat_rotation = False
_C.MODEL.SCAFFOLD.zero_latent = True
_C.MODEL.SCAFFOLD.normal_zero_init = False
# sigmoid, softplus, abs, exp
_C.MODEL.SCAFFOLD.scale_activation = "sigmoid"
_C.MODEL.SCAFFOLD.modify_xyz_offsets = True
_C.MODEL.SCAFFOLD.unit_scale = False
_C.MODEL.SCAFFOLD.unit_scale_multiplier = 10.0
_C.MODEL.SCAFFOLD.offset_scaling = 1.0


_C.MODEL.OPT = CN()
_C.MODEL.OPT.num_steps = 3
_C.MODEL.OPT.add_step_every = 1000
_C.MODEL.OPT.inner_batch_size = 8
_C.MODEL.OPT.inner_downsample = 2
_C.MODEL.OPT.max_grad = 1.0
_C.MODEL.OPT.mask_gradients = False
_C.MODEL.OPT.ssim_mult = 0.2
_C.MODEL.OPT.lpips_mult = 0.1
_C.MODEL.OPT.vgg_mult = 0.0
_C.MODEL.OPT.depth_mult = 0.0
_C.MODEL.OPT.log_depth_mult = 0.0
_C.MODEL.OPT.chamfer_mult = 0.0
_C.MODEL.OPT.occ_mult = 0.0
_C.MODEL.OPT.output_scale = 0.1
_C.MODEL.OPT.scale_gamma = 0.7
_C.MODEL.OPT.min_scale = 0.05
_C.MODEL.OPT.timestamp_norm = False
_C.MODEL.OPT.timestamp_random = False
_C.MODEL.OPT.output_norm = True
_C.MODEL.OPT.decoder_update_last_only = False
_C.MODEL.OPT.grad_residual = False
# 16unetA, 16unetC
_C.MODEL.OPT.backbone = "16unetA"

# 2DGS related
_C.MODEL.OPT.dist_mult = 0.0
_C.MODEL.OPT.normal_mult = 0.0
_C.MODEL.OPT.normal_loss_mult = 0.0

# Test-time related
_C.MODEL.OPT.reset_opacity_ft = False
_C.MODEL.OPT.prune_opacity_ft = False


_C.MODEL.INIT = CN()
# 16unet18, 16unet34
_C.MODEL.INIT.model_type = "16unet18"
_C.MODEL.INIT.radius = 4
_C.MODEL.INIT.use_balance_weight = True
_C.MODEL.INIT.dilate_gt = 0
_C.MODEL.INIT.num_dense_blocks = 2
_C.MODEL.INIT.use_mask_loss = False
_C.MODEL.INIT.occ_reg_mult = 0.0
# "opacity+rgb+normal+scale", "opacity+rgb", "opacity"
_C.MODEL.INIT.init_type = "opacity+rgb+normal+scale"
# For ablation study purposes
_C.MODEL.INIT.opt_enable = False
_C.MODEL.INIT.train_geometry_only_steps = 0


_C.MODEL.DENSIFIER = CN()
_C.MODEL.DENSIFIER.enable = True
_C.MODEL.DENSIFIER.model_type = "16unet18"
_C.MODEL.DENSIFIER.num_dense_blocks = 0
_C.MODEL.DENSIFIER.sample_num_per_step = 1000
_C.MODEL.DENSIFIER.sample_num_per_step_eval = 0
_C.MODEL.DENSIFIER.sample_temperature = 1.0
_C.MODEL.DENSIFIER.sample_temperature_decay = 0.999
_C.MODEL.DENSIFIER.sample_temperature_min = 1.0
_C.MODEL.DENSIFIER.sample_threshold = 0
_C.MODEL.DENSIFIER.loss_ignore_exist = False
_C.MODEL.DENSIFIER.dilate_ignore = 0
_C.MODEL.DENSIFIER.ignore_second_last = False
_C.MODEL.DENSIFIER.eval_randomness = False
_C.MODEL.DENSIFIER.recompute_grad = False
# occupancy_as_opacity should be true unless ablation
_C.MODEL.DENSIFIER.occupancy_as_opacity = True
_C.MODEL.DENSIFIER.num_densify_steps_eval = 10
_C.MODEL.DENSIFIER.train_densify_only_steps = 0


_C.MODEL.GSPLAT = CN()
_C.MODEL.GSPLAT.init_num_points = 100_000
_C.MODEL.GSPLAT.init_sh_degree = 0     # Key changes
_C.MODEL.GSPLAT.max_sh_degree = 3
_C.MODEL.GSPLAT.position_lr_init = 0.00016
_C.MODEL.GSPLAT.position_lr_final = 0.0000016
_C.MODEL.GSPLAT.position_lr_delay_mult = 0.01
_C.MODEL.GSPLAT.position_lr_max_steps = 30_000
_C.MODEL.GSPLAT.feature_lr = 0.0025
_C.MODEL.GSPLAT.feature_rest_lr = 0.0025 / 20.0
_C.MODEL.GSPLAT.opacity_lr = 0.05
_C.MODEL.GSPLAT.scaling_lr = 0.005
_C.MODEL.GSPLAT.rotation_lr = 0.001
_C.MODEL.GSPLAT.percent_dense = 0.01
_C.MODEL.GSPLAT.init_opacity = 0.1
_C.MODEL.GSPLAT.lambda_dssim = 0.2
_C.MODEL.GSPLAT.densification_interval = 100
_C.MODEL.GSPLAT.opacity_reset_interval = 3000
_C.MODEL.GSPLAT.densify_from_iter = 500
_C.MODEL.GSPLAT.densify_until_iter = 15_000
_C.MODEL.GSPLAT.densify_grad_threshold = 0.0002
_C.MODEL.GSPLAT.density_control = True

# 2DGS related
_C.MODEL.GSPLAT.lambda_dist = 0.0
_C.MODEL.GSPLAT.lambda_normal = 0.0
_C.MODEL.GSPLAT.normal_start_iter = 7000
_C.MODEL.GSPLAT.dist_start_iter = 3000


def get_default_opt_config():
    cfg = CN()
    # "adam" or "adamw"
    cfg.optimizer_type = "adam"
    cfg.lr = 2e-4
    cfg.weight_decay = 0.0
    cfg.eps = 1e-8
    cfg.beta1 = 0.9
    cfg.beta2 = 0.999

    # If None, no grad clipping
    cfg.max_norm = None

    # sceheduler_type: "none", "exp"
    cfg.scheduler_type = "exp"
    # if lr_final == None, then use lr as lr_final
    cfg.lr_final = None
    cfg.step_size = 1000
    cfg.max_steps = 100000
    cfg.warmup_steps = 0
    cfg.lr_pre_warmup = 1e-6
    cfg.ramp = "consine"
    return cfg.clone()


_C.OPTIMIZER = CN()
_C.OPTIMIZER.model = get_default_opt_config()

_C.OPTIMIZER.densifier = get_default_opt_config()
_C.OPTIMIZER.densifier.lr = 1e-4

_C.OPTIMIZER.decoder = get_default_opt_config()
_C.OPTIMIZER.decoder.lr = 2e-4


def get_cfg_defaults():
    return _C.clone()
