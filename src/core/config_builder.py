from mmcv import Config
from omegaconf import OmegaConf

from utils.utils import register_custom_resolvers

register_custom_resolvers()


def hydra_to_mmcv(cfg: OmegaConf) -> Config:
    """Convert Hydra config to MMCV config.

    Args:
        cfg: The Hydra config to convert.

    Returns:
        The MMCV config.

    """
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # data
    cfg['data'] = cfg.get('data', cfg.get('dataset', {}).get('data', {}))

    # optimizer
    optimizer_cfg = cfg['optimizer']
    optimizer_config = optimizer_cfg.pop('optimizer_config', {})
    cfg['optimizer'] = optimizer_cfg
    cfg['optimizer_config'] = optimizer_config

    # runner / epochs
    cfg['runner'] = cfg['trainer']['runner']
    cfg['total_epochs'] = cfg['runner']['max_epochs']

    # trainer-related
    cfg['evaluation'] = cfg['trainer']['evaluation']
    cfg['checkpoint_config'] = cfg['trainer']['checkpoint_config']

    # logging
    cfg['log_config'] = cfg['logger']['log_config']
    cfg['log_level'] = cfg.get('log_level', 'INFO')

    cfg['model'] = cfg['model']

    # runtime
    cfg['work_dir'] = cfg['work_dir']
    cfg['seed'] = cfg['trainer']['seed']
    cfg['gpu_ids'] = cfg['trainer']['gpu_ids']
    cfg['device'] = cfg['trainer']['device']

    # resume / load
    cfg['load_from'] = cfg.get('load_from')
    cfg['resume_from'] = cfg.get('resume_from')
    cfg['auto_resume'] = cfg.get('auto_resume', False)

    # lr / workflow
    cfg['lr_config'] = cfg.get('lr_config')
    cfg['workflow'] = cfg.get('workflow', [('train', 1)])

    # system
    cfg['dist_params'] = cfg.get('dist_params', {'backend': 'nccl'})
    cfg['opencv_num_threads'] = cfg.get('opencv_num_threads', 0)
    cfg['mp_start_method'] = cfg.get('mp_start_method', 'fork')

    # mmrotate
    cfg['angle_version'] = cfg.get('angle_version', 'le90')

    cfg['model_name'] = cfg.get('model_name')
    cfg['custom_hooks'] = cfg['callbacks']['custom_hooks']

    return Config(cfg)
