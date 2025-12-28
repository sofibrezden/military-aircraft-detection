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
