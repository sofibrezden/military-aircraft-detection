import json
import random
from pathlib import Path

import click
import clip
import cv2
import defusedxml.ElementTree as ET
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from peft import LoraConfig
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

SURFACE_CLASSES = ['concrete runway', 'airport apron', 'asphalt runway', 'grass field', 'urban ground', 'desert ground']
WEATHER_CLASSES = ['clear weather', 'cloudy weather', 'foggy weather', 'rainy weather']
TIME_CLASSES = ['daytime', 'sunset', 'nighttime']


def compute_generation_plan(class_stats: dict[str, int], alpha: float) -> dict[str, int]:
    """Compute the generation plan for the dataset."""
    Nmax = max(class_stats.values())
    plan: dict[str, int] = {}
    for cls, Nc in class_stats.items():
        plan[cls] = int(alpha * (Nmax - Nc))
    return plan


def load_annotation(path: Path) -> tuple[np.ndarray, str]:
    """Load the annotation from XML file."""
    tree = ET.parse(path)
    root = tree.getroot()

    obj = root.find('object')
    if obj is None:
        msg = f'No object found in {path}'
        raise ValueError(msg)

    robndbox = obj.find('robndbox')
    if robndbox is None:
        msg = f'No robndbox found in {path}'
        raise ValueError(msg)

    bbox = np.array(
        [
            [float(robndbox.find('x_left_top').text), float(robndbox.find('y_left_top').text)],
            [float(robndbox.find('x_right_top').text), float(robndbox.find('y_right_top').text)],
            [float(robndbox.find('x_right_bottom').text), float(robndbox.find('y_right_bottom').text)],
            [float(robndbox.find('x_left_bottom').text), float(robndbox.find('y_left_bottom').text)],
        ]
    )

    cls = obj.find('name').text

    return bbox, cls


def extract_aircraft(src: np.ndarray, bbox: np.ndarray, predictor: SamPredictor) -> tuple[np.ndarray, np.ndarray]:
    """Extract the aircraft from the image."""
    predictor.set_image(src)
    box = np.array([bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()])
    masks, _, _ = predictor.predict(box=box)
    mask = masks[0].astype(np.uint8)
    obj = src * mask[:, :, None]
    return obj, mask


def resize_object(obj: np.ndarray, mask: np.ndarray, w_syn: int, h_syn: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize the object to the synthetic size."""
    obj = cv2.resize(obj, (w_syn, h_syn))
    mask = cv2.resize(mask, (w_syn, h_syn), interpolation=cv2.INTER_NEAREST)
    return obj, mask


def rotate_object(obj: np.ndarray, mask: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the object to the synthetic size."""
    h, w = obj.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.rad2deg(theta), 1.0)
    obj = cv2.warpAffine(obj, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return obj, mask


def composite(bg: np.ndarray, obj: np.ndarray, mask: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Composite the object into the background."""
    h, w = obj.shape[:2]
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)

    x2 = x1 + w
    y2 = y1 + h

    if x1 < 0 or y1 < 0 or x2 > bg.shape[1] or y2 > bg.shape[0]:
        return None

    roi = bg[y1:y2, x1:x2]
    mask = mask[:, :, None]
    comp = mask * obj + (1 - mask) * roi
    bg[y1:y2, x1:x2] = comp.astype(np.uint8)

    return bg


def affine_transform(points: np.ndarray, scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    """Apply an affine transformation to a set of points."""
    R = np.array([[scale * np.cos(theta), -scale * np.sin(theta)], [scale * np.sin(theta), scale * np.cos(theta)]])
    points = np.asarray(points) @ R.T
    points[:, 0] += tx
    points[:, 1] += ty
    return points


def classify(image_tensor: torch.Tensor, model, classes: list[str], device: str) -> str:
    """Classify the image into a surface, weather, or time of day class."""
    tokens = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokens)

        logits = image_features @ text_features.T

        pred = logits.softmax(dim=-1).argmax()

    return classes[pred]


def classify_scene(image: np.ndarray, model, preprocess, device: str) -> tuple[str, str, str]:
    """Classify the scene into a surface, weather, or time of day class."""
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    surface = classify(image_tensor, model, SURFACE_CLASSES, device)
    weather = classify(image_tensor, model, WEATHER_CLASSES, device)
    time_of_day = classify(image_tensor, model, TIME_CLASSES, device)

    return surface, weather, time_of_day


def canny_map(image: np.ndarray) -> Image.Image:
    """Create a canny map for the image."""
    edges = cv2.Canny(image, 100, 200)
    edges = np.stack([edges] * 3, axis=2)
    return Image.fromarray(edges)


def compute_class_stats(ann_dir: Path) -> dict[str, int]:
    """Compute the class statistics for the dataset."""
    stats: dict[str, int] = {}
    for ann_path in Path(ann_dir).glob('*.xml'):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls = obj.find('name').text
            stats.setdefault(cls, 0)
            stats[cls] += 1
    return stats


def compute_aircraft_size(ann_dir: Path) -> tuple[float, float]:
    """Compute the median size of aircrafts in the dataset."""
    widths: list[float] = []
    heights: list[float] = []

    for ann_path in Path(ann_dir).glob('*.xml'):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            robndbox = obj.find('robndbox')
            if robndbox is None:
                continue

            bbox = np.array(
                [
                    [float(robndbox.find('x_left_top').text), float(robndbox.find('y_left_top').text)],
                    [float(robndbox.find('x_right_top').text), float(robndbox.find('y_right_top').text)],
                    [float(robndbox.find('x_right_bottom').text), float(robndbox.find('y_right_bottom').text)],
                    [float(robndbox.find('x_left_bottom').text), float(robndbox.find('y_left_bottom').text)],
                ]
            )

            w = np.linalg.norm(bbox[0] - bbox[1])
            h = np.linalg.norm(bbox[1] - bbox[2])

            widths.append(w)
            heights.append(h)

    w_med = float(np.median(widths))
    h_med = float(np.median(heights))
    return w_med, h_med


@click.command()
@click.option('--images-dir', default='data/JPEGImages', type=click.Path(path_type=Path), required=True)
@click.option('--ann-dir', default='data/Annotations/Oriented Bounding Boxes', type=click.Path(path_type=Path), required=True)
@click.option('--sam-checkpoint', default='src/sam_vit_h_4b8939.pth', type=click.Path(path_type=Path), required=True)
@click.option('--lora-path', default='data/lora_inpainting/lora_weights.pth', type=click.Path(path_type=Path), required=True)
@click.option('--out-dir', default='data/generated', type=click.Path(path_type=Path), required=True)
@click.option('--alpha', default=0.7)
def generate_dataset(images_dir: Path, ann_dir: Path, sam_checkpoint: Path, lora_path: Path, out_dir: Path, alpha: float) -> None:
    """Generate a synthetic dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    click.echo('Loading SAM model...')
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    click.echo('Loading CLIP model...')
    clip_model, preprocess = clip.load('ViT-B/32', device=device)

    click.echo('Loading diffusion pipeline...')
    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny', torch_dtype=dtype)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', controlnet=controlnet, torch_dtype=dtype
    )

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'], lora_dropout=0.1)
    pipe.unet.add_adapter(lora_config)

    lora_state_dict = torch.load(lora_path, map_location='cpu')
    pipe.unet.load_state_dict(lora_state_dict, strict=False)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ModuleNotFoundError:
        click.echo('xformers not available, using standard attention')

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing(1)

    if device == 'cuda':
        sam.to('cpu')
        clip_model.to('cpu')
        torch.cuda.empty_cache()
        click.echo('Moved SAM and CLIP to CPU to free GPU memory')

    stats = compute_class_stats(ann_dir)

    plan = compute_generation_plan(stats, alpha)
    w_med, h_med = compute_aircraft_size(ann_dir)
    images = list(images_dir.glob('*.jpg'))
    annotations = list(Path(ann_dir).glob('*.xml'))

    out_img = Path(out_dir) / 'images'
    out_ann = Path(out_dir) / 'annotations'

    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    synthetic_id = 0
    ann_index = {}

    for a in annotations:
        tree = ET.parse(a)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            ann_index.setdefault(cls_name, []).append((a, obj))
    for cls, n_samples in plan.items():
        cls_ann = ann_index[cls]

        for _ in tqdm(range(n_samples), desc=f'Generating {cls}'):
            ann_path, xml_obj = random.choice(cls_ann)

            robndbox = xml_obj.find('robndbox')
            bbox = np.array(
                [
                    [float(robndbox.find('x_left_top').text), float(robndbox.find('y_left_top').text)],
                    [float(robndbox.find('x_right_top').text), float(robndbox.find('y_right_top').text)],
                    [float(robndbox.find('x_right_bottom').text), float(robndbox.find('y_right_bottom').text)],
                    [float(robndbox.find('x_left_bottom').text), float(robndbox.find('y_left_bottom').text)],
                ]
            )

            src_img_path = Path(images_dir) / (ann_path.stem + '.jpg')
            src = cv2.imread(str(src_img_path))

            if device == 'cuda':
                sam.to(device)
            obj, mask = extract_aircraft(src, bbox, predictor)
            if device == 'cuda':
                sam.to('cpu')
                torch.cuda.empty_cache()

            tgt_bg_path = random.choice(images)
            tgt_bg = cv2.imread(str(tgt_bg_path))
            H, W = tgt_bg.shape[:2]

            if device == 'cuda':
                clip_model.to(device)
            surface, weather, time_of_day = classify_scene(tgt_bg, clip_model, preprocess, device)
            if device == 'cuda':
                clip_model.to('cpu')
                torch.cuda.empty_cache()

            prompt = f'a satellite image of {surface}, {weather}, {time_of_day}'
            theta = np.deg2rad(random.uniform(0, 180))

            w_syn = int(random.uniform(0.9 * w_med, 1.1 * w_med))
            h_syn = int(random.uniform(0.9 * h_med, 1.1 * h_med))

            obj, mask = resize_object(obj, mask, w_syn, h_syn)
            obj, mask = rotate_object(obj, mask, theta)

            cx = random.randint(w_syn // 2, W - w_syn // 2)
            cy = random.randint(h_syn // 2, H - h_syn // 2)

            mask_bg = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(mask_bg, (cx, cy), max(w_syn, h_syn), 255, -1)
            control = canny_map(tgt_bg)
            control = control.resize((tgt_bg.shape[1], tgt_bg.shape[0]))

            result = pipe(
                prompt=prompt,
                negative_prompt='aircraft, airplane, jet, helicopter',
                image=Image.fromarray(cv2.cvtColor(tgt_bg, cv2.COLOR_BGR2RGB)),
                mask_image=Image.fromarray(mask_bg),
                control_image=control,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

            result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            result = composite(result, obj, mask, cx, cy)

            if result is None:
                click.echo(f'Failed to generate image for {cls}')
                continue

            bbox_center = bbox.mean(axis=0)

            bbox_norm = bbox - bbox_center

            scale_x = w_syn / np.linalg.norm(bbox[0] - bbox[1])
            scale_y = h_syn / np.linalg.norm(bbox[1] - bbox[2])

            bbox_norm[:, 0] *= scale_x
            bbox_norm[:, 1] *= scale_y

            bbox_new = affine_transform(bbox_norm, 1.0, theta, cx, cy)
            img_name = f'{synthetic_id:06d}.jpg'
            cv2.imwrite(str(out_img / img_name), result)
            ann = {'image': img_name, 'class': cls, 'bbox': bbox_new.tolist()}

            with Path(out_ann / f'{synthetic_id:06d}.json').open('w') as f:
                json.dump(ann, f)

            synthetic_id += 1

            if device == 'cuda' and synthetic_id % 5 == 0:
                torch.cuda.empty_cache()


if __name__ == '__main__':
    generate_dataset()
