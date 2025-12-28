# Military Aircraft Detection (Oriented)

A deep learning project for detecting military aircraft in images using oriented object detection models. This project implements three state-of-the-art rotated object detection architectures: Oriented RCNN, R3Det, and RoITransformer, built on top of the MMRotate framework.

## Features

- **Three Detection Models**: 
  - Oriented RCNN (two-stage detector)
  - R3Det (refined single-stage detector)
  - RoITransformer (region-based transformer)

- **Dataset**: Uses the [Military Aircraft Recognition dataset](https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset) from Kaggle. The dataset must be downloaded to the `data/` folder and converted to DOTA format using the provided conversion script.

- **Training Pipeline**: 
  - Hydra-based configuration management
  - TensorBoard logging
  - Early stopping hooks
  - Using of pre-trained model on DOTA checkpoints

- **Inference Tools**:
  - Command-line inference script
  - Interactive Gradio web interface

- **Evaluation**: Comprehensive mAP evaluation with per-class metrics

## Requirements

- Python 3.10 (see `.python-version`)
- CUDA 11.6 compatible GPU (for training/inference)
- PyTorch 1.12.1
- MMRotate 0.3.4
- MMDetection 2.28.2
- Other dependencies listed in `pyproject.toml`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sofibrezden/military-aircraft-detection.git
cd military-aircraft-detection
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

3. Download pre-trained checkpoints:
   - Pre-trained models are located in `checkpoints/pretrained/`
   - Fine-tuned models are in `checkpoints/finetuned/`

4. Prepare the dataset:
   - Download the [Military Aircraft Recognition dataset](https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset) from Kaggle
   - Extract it to the `data/` folder (should contain `JPEGImages/` and `Annotations/` directories)
   - Convert the dataset to DOTA format:
   ```bash
   python src/utils/converter_to_dota_dataset.py
   ```

## Usage

### Training

Train a model using Hydra configuration:

```bash
# Train Oriented RCNN
python src/train.py --config-name train_oriented_rcnn

# Train R3Det
python src/train.py --config-name train_r3det

# Train RoITransformer
python src/train.py --config-name train_roitrans
```

Training configurations can be customized by editing the respective YAML files in `src/configs/`.

### Inference

#### Command-line Inference

Run inference on a single image:

```bash
python src/inference.py \
    --img /path/to/image.jpg \
    --config src/configs/train_oriented_rcnn.yaml \
    --checkpoint checkpoints/finetuned/oriented_rccn_latest.pth \
    --out-file results/output.jpg \
    --device cuda:0 \
    --score-thr 0.3
```

#### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python src/app.py --server-name 0.0.0.0 --server-port 7860
```

The interface allows you to:
- Upload images for detection
- Select between different models
- Adjust score threshold
- View example predictions with ground truth comparisons

## Project Structure

```
.
├── src/
│   ├── app.py                 # Gradio web interface
│   ├── train.py               # Training script
│   ├── inference.py           # Command-line inference
│   ├── registry.py            # Model/dataset registry
│   ├── configs/               # Hydra configuration files
│   │   ├── train_*.yaml       # Training configs
│   │   ├── model/             # Model architectures
│   │   ├── dataset/           # Dataset configs
│   │   ├── optimizer/         # Optimizer configs
│   │   └── ...
│   ├── core/
│   │   ├── trainer.py         # Training wrapper
│   │   ├── config_builder.py  # Config conversion utilities
│   │   └── hooks/             # Training hooks
│   ├── datasets/
│   │   └── dota_aircraft.py   # Custom dataset class
│   └── utils/
│       ├── utils.py                        # Utility functions and constants
│       ├── converter_to_dota_dataset.py    # Convert XML annotations to DOTA format
│       └── visualize_gt.py                 # Visualize ground truth bounding boxes
├── checkpoints/
│   ├── pretrained/            # Pre-trained model weights
│   └── finetuned/             # Fine-tuned model weights
├── data/                      # Dataset files
├── logs/                      # Training logs and checkpoints
├── outputs/                   # Training output logs
└── pyproject.toml             # Project dependencies
```

## Models

### Oriented RCNN
Two-stage detector with oriented region proposal network (RPN) and rotated ROI head.

### R3Det
Refined single-stage detector with iterative refinement mechanism.

### RoITransformer
Region-based transformer architecture for rotated object detection.

All models use ResNet-50 backbone with FPN neck and are pre-trained on DOTA dataset.

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `src/configs/train_*.yaml`: Main training configurations
- `src/configs/model/*.yaml`: Model architecture definitions
- `src/configs/dataset/dataset.yaml`: Dataset configuration
- `src/configs/optimizer/*.yaml`: Optimizer settings

## Evaluation

Models are evaluated using mean Average Precision (mAP) with rotated bounding boxes. The evaluation includes:
- Overall mAP
- Per-class AP scores
- Per-class recall metrics
- Detection counts

## Examples

Example images with ground truth annotations are available in `src/examples/`. These demonstrate:
- Input images
- Ground truth bounding boxes
- Model predictions

## Acknowledgments

- [MMRotate](https://github.com/open-mmlab/mmrotate) for the detection framework
- [MMDetection](https://github.com/open-mmlab/mmdetection) for the base detection library
- DOTA dataset for pre-trained models

