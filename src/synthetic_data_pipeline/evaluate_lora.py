import json
from pathlib import Path

import click
import clip

# LPIPS
import lpips
import numpy as np
import torch
from PIL import Image

# FID
from pytorch_fid.fid_score import calculate_fid_given_paths
from tqdm import tqdm


def load_captions(jsonl_path: Path) -> list[dict]:
    """Load captions from a JSONL file."""
    data = []
    with jsonl_path.open() as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def compute_clip_score(images_dir: Path, captions_jsonl: Path, model_name: str, device: str) -> float:
    """Compute the CLIP score for a given directory of images and captions."""
    model, preprocess = clip.load(model_name, device=device)
    data = load_captions(captions_jsonl)

    scores = []
    with torch.no_grad():
        for item in tqdm(data, desc='CLIP score'):
            img_path = images_dir / item['file_name']
            if not img_path.exists():
                continue

            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            text = clip.tokenize([item['text']]).to(device)

            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            sim = (image_features @ text_features.T).item()
            scores.append(sim)

    return float(np.mean(scores)) if scores else float('nan')


def compute_lpips(real_dir: Path, gen_dir: Path, device: str) -> float:
    """Compute the LPIPS score for a given directory of real and generated images."""
    loss_fn = lpips.LPIPS(net='alex').to(device)

    real_files = sorted(real_dir.glob('*.png'))
    gen_files = sorted(gen_dir.glob('*.png'))

    scores = []

    with torch.no_grad():
        for r, g in tqdm(zip(real_files, gen_files, desc='LPIPS', strict=False), total=min(len(real_files), len(gen_files))):
            real = Image.open(r).convert('RGB').resize((256, 256))
            gen = Image.open(g).convert('RGB').resize((256, 256))

            real = torch.tensor(np.array(real)).permute(2, 0, 1).float() / 127.5 - 1
            gen = torch.tensor(np.array(gen)).permute(2, 0, 1).float() / 127.5 - 1

            real = real.unsqueeze(0).to(device)
            gen = gen.unsqueeze(0).to(device)

            score = loss_fn(real, gen).item()
            scores.append(score)

    return float(np.mean(scores)) if scores else float('nan')


@click.command()
@click.option(
    '--real-dir', default='data/real', type=click.Path(path_type=Path), required=True, help='Directory with real background images.'
)
@click.option(
    '--gen-dir', default='data/generated', type=click.Path(path_type=Path), required=True, help='Directory with generated images.'
)
@click.option(
    '--captions-json',
    default='data/captions.jsonl',
    type=click.Path(path_type=Path),
    required=True,
    help='JSONL captions file used for generation.',
)
@click.option('--clip-model', default='ViT-B/32', show_default=True)
@click.option('--batch-size', default=50, type=int, show_default=True)
@click.option('--device', default='cpu', help='cuda or cpu (auto if not set)')
@click.option('--compute-lpips/--no-lpips', default=False, show_default=True, help='Compute LPIPS if paired real/gen images are available.')
def main(real_dir: Path, gen_dir: Path, captions_json: Path, clip_model: str, batch_size: int, device: str, compute_lpips: bool) -> None:
    """Evaluate LoRA-generated images with FID, CLIP score and optionally LPIPS."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    click.echo(f'Using device: {device}')

    click.echo('Computing FID...')
    fid = calculate_fid_given_paths([str(real_dir), str(gen_dir)], batch_size, device, dims=2048)

    click.echo('Computing CLIP score...')
    clip_score = compute_clip_score(gen_dir, captions_json, clip_model, device)

    lpips_score = None
    if compute_lpips:
        click.echo('Computing LPIPS...')
        lpips_score = compute_lpips(real_dir, gen_dir, device)

    click.echo('\n===== Evaluation Results =====')
    click.echo(f'FID: {fid:.4f}')
    click.echo(f'CLIP Score: {clip_score:.4f}')

    if lpips_score is not None:
        click.echo(f'LPIPS: {lpips_score:.4f}')


if __name__ == '__main__':
    main()
