import argparse
import tempfile
from pathlib import Path

import gradio as gr
import hydra
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmrotate.models.detectors import RotatedBaseDetector

import registry  # noqa: F401
from core.config_builder import hydra_to_mmcv
from utils.utils import PALETTE

MODELS = {
    'Oriented RCNN': {'config': 'src/configs/train_oriented_rcnn.yaml', 'checkpoint': 'checkpoints/finetuned/oriented_rccn_latest.pth'},
    'R3Det': {'config': 'src/configs/train_r3det.yaml', 'checkpoint': 'checkpoints/finetuned/r3det_latest.pth'},
    'RoITransformer': {'config': 'src/configs/train_roitrans.yaml', 'checkpoint': 'checkpoints/finetuned/roitrans_latest.pth'},
}

EXAMPLE_INPUTS = ['src/examples/01_input.jpg', 'src/examples/02_input.jpg', 'src/examples/03_input.jpg']
EXAMPLE_GTS = ['src/examples/01_gt.jpg', 'src/examples/02_gt.jpg', 'src/examples/03_gt.jpg']

DEVICE = 'cuda:0'
_MODEL_CACHE = {}


def load_model(model_name: str) -> RotatedBaseDetector:
    """Load a model from the cache or from the filesystem."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    cfg_path = Path(MODELS[model_name]['config']).resolve()
    ckpt_path = Path(MODELS[model_name]['checkpoint']).resolve()

    with hydra.initialize_config_dir(config_dir=str(cfg_path.parent), version_base='1.3', job_name=f'gradio_{model_name}'):
        cfg = hydra.compose(config_name=cfg_path.stem)

    mmcv_cfg = hydra_to_mmcv(cfg)
    model = init_detector(mmcv_cfg, str(ckpt_path), device=DEVICE)

    _MODEL_CACHE[model_name] = model
    return model


def run_inference(image_path: str, model_name: str, score_thr: float) -> str:
    """Run inference on an image."""
    model = load_model(model_name)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        out_path = tmp.name

    result = inference_detector(model, image_path)
    show_result_pyplot(model, image_path, result, score_thr=score_thr, palette=PALETTE, out_file=out_path)

    return out_path


def generate_example_predictions() -> list[str]:
    """Generate example predictions."""
    return [run_inference(img, next(iter(MODELS.keys())), 0.3) for img in EXAMPLE_INPUTS]


EXAMPLE_PREDICTIONS = generate_example_predictions()


def render_demo() -> None:
    """Render the demo."""
    with gr.Blocks(title='Military Aircraft Detection (Oriented)') as demo:
        gr.Markdown('## Military Aircraft Detection (Oriented)')
        gr.Markdown(
            'Upload an image or select an example below. The bottom section shows Input, Ground Truth, and Prediction side by side.'
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type='filepath', label='Input image', height=350)
                model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), value=next(iter(MODELS.keys())), label='Select model')
                score_thr = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label='Score threshold')
                run_btn = gr.Button('Run inference')

            with gr.Column():
                output_image = gr.Image(label='Prediction', height=350)
            run_btn.click(fn=run_inference, inputs=[input_image, model_dropdown, score_thr], outputs=output_image)

        gr.Markdown('## Examples')
        with gr.Row():
            gr.Markdown('### Input')
            gr.Markdown('### Ground Truth')
            gr.Markdown('### Predicted (Score threshold: 0.3, OrientedRCNN)')

        for inp, gt, pred in zip(EXAMPLE_INPUTS, EXAMPLE_GTS, EXAMPLE_PREDICTIONS, strict=True):
            with gr.Row():
                gr.Image(inp, height=250, show_label=False)
                gr.Image(gt, height=250, show_label=False)
                gr.Image(pred, height=250, show_label=False)

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server-port', type=int, default=7860, help='Server port')
    args = parser.parse_args()
    demo = render_demo()
    demo.launch(server_name=args.server_name, server_port=args.server_port)
