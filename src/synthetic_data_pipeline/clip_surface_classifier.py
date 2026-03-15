import json
from pathlib import Path

import click
import clip
import torch
from PIL import Image
from tqdm import tqdm

SURFACE_CLASSES = ['concrete runway', 'airport apron', 'asphalt runway', 'grass field', 'urban ground', 'desert ground']
WEATHER_CLASSES = ['clear weather', 'cloudy weather', 'foggy weather', 'rainy weather']
TIME_CLASSES = ['daytime', 'sunset', 'nighttime']


def classify(image_tensor: torch.Tensor, model, classes: list[str], device: str) -> str:
    """Classify the image into a surface, weather, or time of day class."""
    tokens = clip.tokenize(classes).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(tokens)

        logits = image_features @ text_features.T
        pred = logits.softmax(dim=-1).argmax()

    return classes[pred]


@click.command()
@click.option('--crops-dir', default='data/crops', required=True)
@click.option('--out-json', default='data/crops/captions.jsonl', required=True)
@click.option('--model', default='ViT-B/32')
def generate_captions(crops_dir: str, out_json: str, model: str) -> None:
    """Generate captions for the crops."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(model, device=device)
    files = sorted(Path(crops_dir).iterdir())
    with Path(out_json).open('w') as f:
        for file in tqdm(files):
            image = Image.open(file).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            surface = classify(image_tensor, model, SURFACE_CLASSES, device)
            weather = classify(image_tensor, model, WEATHER_CLASSES, device)
            time_of_day = classify(image_tensor, model, TIME_CLASSES, device)

            caption = f'a satellite image of {surface}, {weather}, {time_of_day}'
            item = {'file_name': file.name, 'text': caption}
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    generate_captions()
