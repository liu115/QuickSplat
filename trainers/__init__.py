from .phase1_trainer import Phase1Trainer
from .phase2_trainer import Phase2Trainer


trainer_mapping = {
    "phase1": Phase1Trainer,
    "phase2": Phase2Trainer,
}


def get_trainer_class(config):
    if config.trainer not in trainer_mapping:
        raise ValueError(f"Trainer {config.trainer} not found")
    print(f"Using trainer {trainer_mapping[config.trainer]}")
    return trainer_mapping[config.trainer]


def get_trainer(config, ckpt_path, local_rank, world_size, mode):
    trainer_class = get_trainer_class(config)
    return trainer_class(config, local_rank, world_size, ckpt_path, mode)
