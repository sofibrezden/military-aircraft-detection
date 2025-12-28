from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmrotate.apis import train_detector


class MMRotateTrainer:
    """Trainer for MMRotate models."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.datasets = []
        self.model = None

    def _set_seed(self) -> None:
        set_random_seed(self.cfg.seed, deterministic=False)

    def _build_datasets(self) -> None:
        self.datasets = [build_dataset(self.cfg.data.train)]

    def _build_model(self) -> None:
        self.model = build_detector(self.cfg.model)

        # mmrotate requirement
        self.model.CLASSES = self.datasets[0].CLASSES

    def fit(self) -> None:
        """Run training."""
        self._set_seed()
        self._build_datasets()
        self._build_model()

        train_detector(self.model, self.datasets, self.cfg, distributed=False, validate=True)
