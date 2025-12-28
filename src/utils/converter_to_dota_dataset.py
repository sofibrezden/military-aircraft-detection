"""Convert XML annotations to DOTA format with correct folder structure."""

import shutil
from pathlib import Path

import numpy as np
from defusedxml.ElementTree import parse as ET_parse
from tqdm import tqdm

DATA_ROOT = Path('./data')
XML_DIR = DATA_ROOT / 'Annotations' / 'Oriented Bounding Boxes'
IMAGE_DIR = DATA_ROOT / 'JPEGImages'

IMAGESETS_DIR = DATA_ROOT / 'ImageSets' / 'Main'

OUT_ROOT = Path('./data/DOTA_annotations')
OUT_IMAGES = OUT_ROOT / 'images'
OUT_TRAIN = OUT_ROOT / 'train'
OUT_VAL = OUT_ROOT / 'val'


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convert_xml_to_dota() -> bool:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    # Get train and val IDs to know which files we need
    train_ids = set((IMAGESETS_DIR / 'train.txt').read_text().splitlines())
    val_ids = set((IMAGESETS_DIR / 'test.txt').read_text().splitlines())
    needed_ids = train_ids | val_ids

    xml_files = [XML_DIR / f'{img_id}.xml' for img_id in needed_ids if (XML_DIR / f'{img_id}.xml').exists()]

    converted = 0

    for xml_file in tqdm(xml_files, desc='Converting XML'):
        tree = ET_parse(str(xml_file))
        root = tree.getroot()

        anns = []

        for obj in root.findall('object'):
            cls = obj.findtext('name')
            diff = obj.findtext('difficult', default='0')

            robnd = obj.find('robndbox')
            if robnd is None:
                continue

            try:
                coords = [
                    float(robnd.findtext('x_left_top')),
                    float(robnd.findtext('y_left_top')),
                    float(robnd.findtext('x_right_top')),
                    float(robnd.findtext('y_right_top')),
                    float(robnd.findtext('x_right_bottom')),
                    float(robnd.findtext('y_right_bottom')),
                    float(robnd.findtext('x_left_bottom')),
                    float(robnd.findtext('y_left_bottom')),
                ]

                poly = np.array(coords).reshape(4, 2)
                if polygon_area(poly) < 4:
                    continue

                anns.append(' '.join(f'{c:.1f}' for c in coords) + f' {cls} {diff}')

            except Exception:
                continue

        if anns:
            out_txt = OUT_ROOT / f'{xml_file.stem}.txt'
            out_txt.write_text('\n'.join(anns))
            converted += 1

    print(f'Converted {converted} files')
    return True


def create_splits() -> None:
    train_ids = (IMAGESETS_DIR / 'train.txt').read_text().splitlines()
    val_ids = (IMAGESETS_DIR / 'test.txt').read_text().splitlines()

    print(f'Train: {len(train_ids)} | Val: {len(val_ids)}')

    train_copied = 0
    for img_id in train_ids:
        src = OUT_ROOT / f'{img_id}.txt'
        if src.exists():
            shutil.copy(src, OUT_TRAIN / src.name)
            train_copied += 1

    val_copied = 0
    for img_id in val_ids:
        src = OUT_ROOT / f'{img_id}.txt'
        if src.exists():
            shutil.copy(src, OUT_VAL / src.name)
            val_copied += 1

    print(f'Train/Val annotations ready: {train_copied} train, {val_copied} val files')


def copy_images() -> None:
    # Get train and val IDs to know which images we need
    train_ids = (IMAGESETS_DIR / 'train.txt').read_text().splitlines()
    val_ids = (IMAGESETS_DIR / 'test.txt').read_text().splitlines()
    needed_ids = set(train_ids + val_ids)

    # Only copy images that we actually need
    imgs = []
    for img_id in needed_ids:
        img_path = IMAGE_DIR / f'{img_id}.jpg'
        if img_path.exists():
            imgs.append(img_path)

    for img in tqdm(imgs, desc='Copying images'):
        shutil.copy(img, OUT_IMAGES / img.name)

    print(f'Copied {len(imgs)} images')


if __name__ == '__main__':
    print('Creating DOTA dataset structure')
    convert_xml_to_dota()
    create_splits()
    copy_images()
    print('\nDONE! Dataset is MMRotate-ready')
