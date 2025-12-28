import logging
import os

import numpy as np
from mmrotate.core import eval_rbbox_map
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from utils.utils import CLASSES

@ROTATED_DATASETS.register_module()
class DOTAAircraftDataset(DOTADataset):
    """Military Aircraft dataset in DOTA format."""
    # required by mmrotate
    CLASSES = CLASSES

    def load_annotations(self, ann_folder: str) -> list[dict]:
        """Load annotations from DOTA format TXT files."""
        data_infos = super().load_annotations(ann_folder)

        for data_info in data_infos:
            if data_info['filename'].endswith('.png'):
                data_info['filename'] = data_info['filename'].replace('.png', '.jpg')

        return data_infos

    def evaluate(
        self,
        results: list[dict],
        metric: str = 'mAP',
        logger: logging.Logger | None = None,
        _: tuple[int, int, int] = (100, 300, 1000),
        iou_thr: float = 0.5,
        scale_ranges: tuple[float, float] | None = None,
        nproc: int = 4,
    ) -> dict:
        """Evaluate the dataset.

        Args:
            results: The results to evaluate.
            metric: The metric to evaluate.
            logger: The logger to use.
            _: The proposal numbers to use.
            iou_thr: The IoU threshold to use.
            scale_ranges: The scale ranges to use.
            nproc: The number of processes to use.

        Returns:
            The evaluation results.

        """
        nproc = min(nproc, os.cpu_count())

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]

        if metric != 'mAP':
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}

        assert isinstance(iou_thr, float)

        mean_ap, eval_results_detailed = eval_rbbox_map(
            results, annotations, scale_ranges=scale_ranges, iou_thr=iou_thr, dataset=self.CLASSES, logger=logger, nproc=nproc
        )

        eval_results['mAP'] = mean_ap
        # Per-class metrics flattened
        for class_idx, class_result in enumerate(eval_results_detailed):
            class_name = self.CLASSES[class_idx]
            prefix = f'per_class/{class_name}'
            if isinstance(class_result, dict):
                for k, v in class_result.items():
                    if k == 'ap':
                        eval_results[f'{prefix}/ap'] = float(v)

                    elif k == 'recall' and isinstance(v, np.ndarray) and v.size > 0:
                        eval_results[f'{prefix}/recall_max'] = float(v.max())

                    elif k in ('num_gts', 'num_dets'):
                        eval_results[f'{prefix}/{k}'] = int(v)

            elif isinstance(class_result, (list, tuple, np.ndarray)):
                if len(class_result) > 0:
                    eval_results[f'{prefix}/ap'] = float(class_result[0])
                if len(class_result) > 1:
                    eval_results[f'{prefix}/num_dets'] = int(class_result[1])
                if len(class_result) > 2:
                    eval_results[f'{prefix}/num_gts'] = int(class_result[2])
                if len(class_result) > 3:
                    eval_results[f'{prefix}/recall'] = float(class_result[3])

            else:
                # Only AP is available
                eval_results[f'{prefix}/ap'] = float(class_result)

        if logger is not None:
            logger.info('\nPer-class AP:')
            for class_name in self.CLASSES:
                key = f'per_class/{class_name}/ap'
                if key in eval_results:
                    logger.info(f'{class_name}: {eval_results[key]:.4f}')

        return eval_results
