import json
from pathlib import Path

import click
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class LoraDataset(Dataset):
    """Dataset for training the LoRA model."""

    def __init__(self, images_dir: str, captions_json: str, tokenizer: AutoTokenizer, size: int = 512) -> None:

        self.images_dir = Path(images_dir)
        self.size = size
        self.tokenizer = tokenizer
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5], [0.5])])

        with Path(captions_json).open() as f:
            self.data = [json.loads(x) for x in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:

        item = self.data[idx]

        img = Image.open(self.images_dir / item['file_name']).convert('RGB')
        pixel_values = self.transform(img)

        text = item['text']
        text_inputs = self.tokenizer(
            text, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt'
        )

        return {'pixel_values': pixel_values, 'input_ids': text_inputs.input_ids.squeeze(0)}


@click.command()
@click.option('--dataset-dir', default='data/crops', required=True)
@click.option('--captions-json', default='data/crops/captions.jsonl', required=True)
@click.option('--output-dir', default='data/lora_inpainting', required=True)
@click.option('--lr', default=1e-4)
@click.option('--epochs', default=5)
@click.option('--batch-size', default=4)
@click.option('--rank', default=8)
def train(dataset_dir: str, captions_json: str, output_dir: str, lr: float, epochs: int, batch_size: int, rank: int) -> None:
    """Train the LoRA model for inpainting."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    ).to(device)

    tokenizer = pipe.tokenizer
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    vae.requires_grad_(requires_grad=False)
    text_encoder.requires_grad_(requires_grad=False)

    lora_config = LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'], lora_dropout=0.1)

    unet.add_adapter(lora_config)

    dataset = LoraDataset(dataset_dir, captions_json, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    unet.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_data in progress_bar:
            pixel_values = batch_data['pixel_values'].to(device, dtype=pipe.dtype)
            input_ids = batch_data['input_ids'].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                encoder_hidden_states = text_encoder(input_ids)[0]

            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=device)
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(loader)
        click.echo(f'Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(lora_state_dict, output_path / 'lora_weights.pth')

    click.echo(f'LoRA weights saved to {output_path / "lora_weights.pth"}')


if __name__ == '__main__':
    train()
