"""Inference script for military aircraft detection.

Example usage:
python src/inference.py \
    --img /root/diploma/data/JPEGImages/27.jpg \
    --config src/configs/train_oriented_rcnn.yaml \
    --checkpoint logs/oriented_rcnn_aircraft/latest.pth\
    --out-file results/27.jpg \
    --device cuda:0 \
    --palette dota \
    --score-thr 0.3
"""

from argparse import ArgumentParser
from pathlib import Path

import hydra
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import registry  # noqa: F401
from core.config_builder import hydra_to_mmcv


def parse_args() -> ArgumentParser:
    """Parse the arguments."""
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization',
    )
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    return parser.parse_args()


def main(args: ArgumentParser) -> None:
    """Run inference on a single image."""
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base='1.3', job_name='inference'):
        cfg = hydra.compose(config_name=config_path.stem)

    mmcv_cfg = hydra_to_mmcv(cfg)
    model = init_detector(mmcv_cfg, args.checkpoint, device=args.device)
    result = inference_detector(model, args.img)
    show_result_pyplot(model, args.img, result, palette=args.palette, score_thr=args.score_thr, out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
