"""Visualize ground truth oriented bounding boxes on images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from defusedxml.ElementTree import parse as ET_parse
from loguru import logger

"""
Example usage:
python src/utils/visualize_gt.py
--image-dir data/JPEGImages
--annotation-dir data/Annotations/Oriented Bounding Boxes
--output-dir src/examples
--image-ids 1 2 3
"""


def parse_xml_annotations(xml_path: Path) -> list[tuple[np.ndarray, str, str]]:
    """Parse XML annotations and extract oriented bounding boxes.

    Args:
        xml_path: Path to XML annotation file

    Returns:
        List of tuples: (polygon_coords, class_name, difficult)
        polygon_coords: numpy array of shape (4, 2) with corner coordinates

    """
    tree = ET_parse(str(xml_path))
    root = tree.getroot()

    annotations = []

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

            # Reshape to (4, 2) - 4 corners, each with (x, y)
            poly = np.array(coords).reshape(4, 2).astype(np.int32)
            annotations.append((poly, cls, diff))

        except Exception as e:
            logger.warning(f'Failed to parse object in {xml_path}: {e}')
            continue

    return annotations


def visualize_gt_boxes(
    image_path: Path,
    xml_path: Path,
    output_path: Path,
    box_color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    *,
    show_labels: bool = True,
) -> None:
    """Visualize ground truth oriented bounding boxes on an image.

    Args:
        image_path: Path to input image
        xml_path: Path to XML annotation file
        output_path: Path to save visualized image
        box_color: BGR color tuple for boxes (default: red)
        thickness: Line thickness for boxes
        show_labels: Whether to show class labels on boxes

    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f'Could not read image: {image_path}')
        return

    annotations = parse_xml_annotations(xml_path)

    for poly, cls, diff in annotations:
        cv2.polylines(img, [poly], isClosed=True, color=box_color, thickness=thickness)

        if show_labels:
            x_min = int(poly[:, 0].min())
            y_min = int(poly[:, 1].min())

            label = f'{cls}'
            if diff == '1':
                label += ' (difficult)'
            cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)
    logger.info(f'Saved visualized image to: {output_path}')


def main() -> None:
    """Run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize ground truth boxes on images')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--annotation-dir', type=str, required=True, help='Directory containing XML annotation files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save visualized images')
    parser.add_argument('--image-ids', type=str, nargs='+', help='Specific image IDs to process (without extension)')

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_ids:
        image_ids = args.image_ids
    else:
        xml_files = list(annotation_dir.glob('*.xml'))
        image_ids = [xml.stem for xml in xml_files]

    for img_id in image_ids:
        image_path = image_dir / f'{img_id}.jpg'
        xml_path = annotation_dir / f'{img_id}.xml'

        if not image_path.exists():
            logger.warning(f'Image not found: {image_path}')
            continue

        if not xml_path.exists():
            logger.warning(f'Annotation not found: {xml_path}')
            continue

        output_filename = f'{img_id}_gt.jpg'
        output_path = output_dir / output_filename

        try:
            visualize_gt_boxes(image_path, xml_path, output_path)
        except Exception as e:
            logger.error(f'Error processing {img_id}: {e}')


if __name__ == '__main__':
    main()
