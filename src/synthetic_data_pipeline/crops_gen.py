from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import parse_xml


def compute_aircraft_size(images_dir: str, ann_dir: str) -> tuple[int, int]:
    """Compute the median size of aircrafts in the dataset."""
    widths = []
    heights = []
    for img_path in Path(images_dir).iterdir():
        if not img_path.name.endswith('.jpg'):
            continue
        xml_path = Path(ann_dir) / img_path.name.replace('.jpg', '.xml')
        if not xml_path.exists():
            continue
        polygons = parse_xml(xml_path)
        for poly in polygons:
            w = np.linalg.norm(poly[0] - poly[1])
            h = np.linalg.norm(poly[1] - poly[2])
            widths.append(w)
            heights.append(h)
    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    return median_w, median_h


@click.command()
@click.option('--images-dir', default='data/JPEGImages', required=True)
@click.option('--ann-dir', default='data/Annotations/Oriented Bounding Boxes', required=True)
@click.option('--out-dir', default='data/crops', required=True)
@click.option('--crop-res', default=512, type=int)
@click.option('--aircraft-threshold', default=0.02, type=float)
@click.option('--scale-factor', default=3, type=int)
def generate_crops(
    images_dir: str, ann_dir: str, out_dir: str = 'data/crops', crop_res: int = 512, aircraft_threshold: float = 0.02, scale_factor: int = 3
) -> None:
    """Generate crops from the dataset."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    median_w, median_h = compute_aircraft_size(images_dir, ann_dir)
    base_size = int(max(median_w, median_h) * scale_factor)

    click.echo(f'Median aircraft size: {median_w} {median_h}')
    click.echo(f'Base crop size: {base_size}')

    counter = 0
    for img_path in tqdm(Path(images_dir).iterdir()):
        if not img_path.name.endswith('.jpg'):
            continue

        xml_path = Path(ann_dir) / img_path.name.replace('.jpg', '.xml')
        if not xml_path.exists():
            continue

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        polygons = parse_xml(xml_path)
        aircraft_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(aircraft_mask, [poly], 255)
        if h < base_size or w < base_size:
            click.echo(f'Image {img_path.name} is too small, skipping')
            continue

        step = base_size // 2
        for y in range(0, h - base_size + 1, step):
            for x in range(0, w - base_size + 1, step):
                crop_mask = aircraft_mask[y : y + base_size, x : x + base_size]
                aircraft_ratio = np.sum(crop_mask > 0) / (base_size * base_size)
                if aircraft_ratio > aircraft_threshold:
                    continue

                crop = image[y : y + base_size, x : x + base_size]
                crop = cv2.resize(crop, (crop_res, crop_res))
                out_path = Path(out_dir) / f'{counter:06d}.png'
                cv2.imwrite(str(out_path), crop)

                counter += 1

    click.echo(f'Total crops: {counter}')


if __name__ == '__main__':
    generate_crops()
