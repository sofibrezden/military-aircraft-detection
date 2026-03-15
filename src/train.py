from pathlib import Path

import click
from hydra import compose, initialize_config_dir

import registry  # noqa: F401
from core.config_builder import hydra_to_mmcv
from core.trainer import MMRotateTrainer


@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice(['train_r3det', 'train_roitrans', 'train_oriented_rcnn'], case_sensitive=False),
    default='train_r3det',
    help='Configuration file to use for training',
)
def main(config: str) -> None:
    """Train the model.

    Args:
        config: Name of the configuration file to use.

    Returns:
        None.

    """
    config_dir = str(Path(__file__).parent / 'configs')

    with initialize_config_dir(config_dir=config_dir, version_base='1.3'):
        cfg = compose(config_name=config)
        mmcv_cfg = hydra_to_mmcv(cfg)
        trainer = MMRotateTrainer(cfg=mmcv_cfg)
        trainer.fit()


if __name__ == '__main__':
    main()
