import hydra
from omegaconf import OmegaConf

import registry  # noqa: F401
from core.config_builder import hydra_to_mmcv
from core.trainer import MMRotateTrainer


@hydra.main(config_path='configs', config_name='train_r3det', version_base='1.3')
def main(cfg: OmegaConf) -> None:
    """Train the model.

    Args:
        cfg: The configuration object.

    Returns:
        None.

    """
    mmcv_cfg = hydra_to_mmcv(cfg)
    trainer = MMRotateTrainer(cfg=mmcv_cfg)
    trainer.fit()


if __name__ == '__main__':
    main()
