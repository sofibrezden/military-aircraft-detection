import defusedxml.ElementTree as ET
import numpy as np
from omegaconf import OmegaConf

# palette for 20 classes
PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (255, 215, 0),
    (75, 0, 130),
    (255, 69, 0),
    (0, 191, 255),
    (154, 205, 50),
]

CLASSES = (
        'A1',
        'A2',
        'A3',
        'A4',
        'A5',
        'A6',
        'A7',
        'A8',
        'A9',
        'A10',
        'A11',
        'A12',
        'A13',
        'A14',
        'A15',
        'A16',
        'A17',
        'A18',
        'A19',
        'A20',
)

def register_custom_resolvers() -> None:
    """Register custom resolvers for OmegaConf."""
    OmegaConf.register_new_resolver('img_scale', lambda w, h=None: (w, w) if h is None else (w, h), replace=True)

def parse_xml(xml_path: str) -> list[np.ndarray]:
    """Parse XML annotations and return polygons."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons = []

    for obj in root.findall("object"):

        robndbox = obj.find("robndbox")

        pts = [
            [float(robndbox.find("x_left_top").text),
             float(robndbox.find("y_left_top").text)],

            [float(robndbox.find("x_right_top").text),
             float(robndbox.find("y_right_top").text)],

            [float(robndbox.find("x_right_bottom").text),
             float(robndbox.find("y_right_bottom").text)],

            [float(robndbox.find("x_left_bottom").text),
             float(robndbox.find("y_left_bottom").text)]
        ]

        polygons.append(np.array(pts, dtype=np.int32))

    return polygons
