import csv
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import hydra
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from mmdet.apis import inference_detector, init_detector
import registry  # noqa: F401
from core.config_builder import hydra_to_mmcv

DEVICE = 'cpu'
_MODEL_CACHE = {}


def load_aircraft_data(csv_path: Path = Path('src/examples/aircraft_infos.csv')) -> tuple[dict, dict]:
    """Load aircraft data from CSV file."""
    class_mapping: dict[str, str] = {}
    aircraft_info: dict[str, dict] = {}

    with csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_mapping[row['class_id']] = row['name']
            aircraft_info[row['name']] = {
                'country': row['country'],
                'role': row['role'],
                'year': int(row['year']),
                'speed': row['speed'],
                'range': row['range'],
                'crew': int(row['crew']),
                'fleet': int(row['fleet']),
                'desc': row['desc'],
                'img': row['img'],
            }

    return class_mapping, aircraft_info


CLASS_MAPPING, AIRCRAFT_INFO = load_aircraft_data()

BASE = 'https://huggingface.co/spaces/sofibrezden/Military-Aircraft-Detection/resolve/main'

EXAMPLES = [
    (f'{BASE}/examples/01_input.jpg', 0.3, f'{BASE}/examples/01_pred.jpg', f'{BASE}/examples/01_gt.jpg'),
    (f'{BASE}/examples/02_input.jpg', 0.5, f'{BASE}/examples/02_pred.jpg', f'{BASE}/examples/02_gt.jpg'),
    (f'{BASE}/examples/03_input.jpg', 0.7, f'{BASE}/examples/03_pred.jpg', f'{BASE}/examples/03_gt.jpg'),
]


def download_image(url: str) -> str:
    """Download image from URL."""
    response = requests.get(url, timeout=30)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name


def build_examples_table() -> str:
    """Build examples table."""
    rows = ''
    for i, (inp, thr, pred, gt) in enumerate(EXAMPLES):
        rows += f"""
        <tr onclick="
            setTimeout(() => {{
                const el = document.querySelector('#example_trigger textarea');
                if (el) {{
                    el.value = '{i}';
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }}, 50);
        " style="cursor:pointer; transition:0.2s">
            <td><img src="{inp}" width="200"></td>
            <td>{thr}</td>
            <td><img src="{pred}" width="200"></td>
            <td><img src="{gt}" width="200"></td>
        </tr>
        """

    return f"""
    <table style="width:100%; margin-top:20px; border-collapse:collapse">
        <tr>
            <th>Input</th>
            <th>Confidence</th>
            <th>Prediction</th>
            <th>GT</th>
        </tr>
        {rows}
    </table>
    """


def build_kpi(total: int, unique: int, us: int, ru: int, avg: float) -> str:
    """Build KPI table."""
    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px">
        <div class="kpi">
            <div class="kpi-value">{total}</div>
            <div class="kpi-label">Detected</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{unique}</div>
            <div class="kpi-label">Types</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{us}</div>
            <div class="kpi-label">USA</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">{ru}</div>
            <div class="kpi-label">rus</div>
        </div>
    </div>
    """


def load_model():
    """Load model."""
    if 'model' in _MODEL_CACHE:
        return _MODEL_CACHE['model']

    ckpt = hf_hub_download(repo_id='sofibrezden/diploma', filename='oriented_rccn_latest.pth')
    cfg_path = Path('src/configs/train_oriented_rcnn.yaml').resolve()

    with hydra.initialize_config_dir(config_dir=str(cfg_path.parent), version_base='1.3'):
        cfg = hydra.compose(config_name=cfg_path.stem)

    model = init_detector(hydra_to_mmcv(cfg), ckpt, device=DEVICE)
    _MODEL_CACHE['model'] = model
    return model


def draw_boxes(img: np.ndarray, result: list[np.ndarray], classes: list[str], thr: float) -> tuple[str, list[str]]:
    """Draw boxes on image."""
    img = img.copy()
    pred_classes = []

    for i, dets in enumerate(result):
        for det in dets:
            score = det[-1]
            if score < thr:
                continue
            cx, cy, w, h, angle = det[:5]
            rect = ((cx, cy), (w, h), angle * 180 / np.pi)
            box = cv2.boxPoints(rect).astype(int)

            name = CLASS_MAPPING.get(classes[i], classes[i])
            pred_classes.append(name)

            cv2.polylines(img, [box], isClosed=True, color=(255, 255, 255), thickness=2)
            label = f'{name} {score:.2f}'
            (font_w, font_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            x, y = box[0]
            y = max(y, font_h + 5)

            cv2.rectangle(img, (x, y - font_h - 4), (x + font_w + 4, y), (0, 0, 0), -1)
            cv2.putText(img, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        return tmp.name, list(set(pred_classes))


def build_chart(us: int, ru: int) -> str:
    """Build chart."""
    max_val = max(us, ru, 1)

    max_height = 120
    us_h = int((us / max_val) * max_height)
    ru_h = int((ru / max_val) * max_height)

    us_zero = us == 0
    ru_zero = ru == 0

    if us_zero:
        us_h = 10
    if ru_zero:
        ru_h = 10

    return f"""
    <div style="margin-top:0px; width:100%;">
        <div style="font-size:16px; font-weight:600; margin-bottom:10px; opacity:0.8">
        Estimated number of such aircraft in national fleets
        </div>
        <style>
            .chart-wrapper {{
                width:100%;
                background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                border-radius:14px;
                padding:20px;
                border:1px solid rgba(255,255,255,0.05);
            }}
            .chart-inner {{
                display:flex;
                justify-content:center;
                align-items:flex-end;
                gap:80px;
                height:180px;
            }}
            .bar {{
                width:60px;
                height:0px;
                border-radius:12px 12px 4px 4px;

                display:flex;
                align-items:center;
                justify-content:center;

                position:relative;
                overflow:hidden;
            }}
            .bar-value {{
                color:white;
                font-size:14px;
                font-weight:600;
                z-index:2;
            }}
            .bar {{
                width:60px;
                border-radius:12px 12px 4px 4px;
                display:flex;
                align-items:center;
                justify-content:center;
                position:relative;
            }}
            .bar-value {{
                color:white;
                font-size:14px;
                font-weight:600;
                text-shadow: 0 2px 4px rgba(0,0,0,0.6);
            }}
            .bar-value.outside {{
                position:absolute;
                top:-18px;
                color:#e5e7eb;
            }}
            .us {{
                background: linear-gradient(180deg, #3b82f6, #1e3a8a);
                box-shadow: 0 8px 20px rgba(59,130,246,0.25);
                animation: growUS 0.9s ease forwards;
            }}
            .ru {{
                background: linear-gradient(180deg, #ef4444, #7f1d1d);
                box-shadow: 0 8px 20px rgba(239,68,68,0.25);
                animation: growRU 0.9s ease forwards;
            }}
            @keyframes growUS {{
                from {{ height:0px; }}
                to {{ height:{us_h}px; }}
            }}
            @keyframes growRU {{
                from {{ height:0px; }}
                to {{ height:{ru_h}px; }}
            }}
            .kpi {{
                background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 12px;
                text-align: center;
            }}

            .kpi-value {{
                font-size: 20px;
                font-weight: 600;
            }}

            .kpi-label {{
                font-size: 12px;
                opacity: 0.7;
            }}
        </style>
        <div class="chart-wrapper">
            <div class="chart-inner">
                <div>
                    <div class="bar us">
                        <span class="bar-value {'outside' if us_zero else ''}">{us}</span>
                    </div>
                    <div>USA</div>
                </div>
                <div>
                    <div class="bar ru">
                        <span class="bar-value {'outside' if ru_zero else ''}">{ru}</span>
                    </div>
                    <div>rus</div>
                </div>
            </div>
        </div>
    </div>
    """


def build_table(pred: list[str]) -> str:
    """Build table."""
    rows = ''
    for p in pred:
        info = AIRCRAFT_INFO.get(p, {})
        rows += f"""
        <tr>
            <td><img src="{info.get('img', '')}" width="250"></td>
            <td>{p}</td>
            <td>{info.get('country', '-')}</td>
            <td>{info.get('role', '-')}</td>
            <td>{info.get('fleet', '-')}</td>
            <td>{info.get('desc', '-')}</td>
            <td>{info.get('year', '-')}</td>
            <td>{info.get('speed', '-')}</td>
            <td>{info.get('range', '-')}</td>
            <td>{info.get('crew', '-')}</td>
        </tr>
        """

    return f"""
    <div style="margin-top:35px; font-size:24px; font-weight:600; opacity:0.8">
    Detected aircraft details
    </div>
    <table style="width:100%;margin-top:20px;font-size:12px">
        <tr>
            <th>Image</th><th>Name</th><th>Country</th>
            <th>Role</th><th>Fleet</th><th>Description</th>
            <th>Year</th>
            <th>Speed</th>
            <th>Range</th>
            <th>Crew</th>
        </tr>
        {rows}
    </table>
    """


def run(img_path: str, thr: float) -> tuple[gr.update, gr.update, gr.update, gr.update, gr.update]:
    """Run detection."""
    model = load_model()
    res = inference_detector(model, img_path)

    img = cv2.imread(img_path)
    img_out, pred = draw_boxes(img, res, model.CLASSES, thr)
    filtered_res = []
    for c in res:
        if len(c) > 0:
            filtered_res.append(c[c[:, -1] >= thr])
        else:
            filtered_res.append(c)

    total = sum(len(c) for c in filtered_res)
    unique = len(pred)

    us = 0
    ru = 0
    for p in pred:
        info = AIRCRAFT_INFO.get(p, {})

        if info.get('country') == 'USA':
            us += info.get('fleet', 0)

        elif info.get('country') == 'rus':
            ru += info.get('fleet', 0)

    scores = []
    for c in res:
        if len(c) > 0:
            scores.extend(c[:, -1])

    avg = round(np.mean(scores), 3) if scores else 0
    return (
        gr.update(value=img_out, visible=True),
        gr.update(value=build_kpi(total, unique, us, ru, avg), visible=True),
        gr.update(value=build_chart(us, ru), visible=True),
        gr.update(value=build_table(pred), visible=True),
        gr.update(visible=False),
    )


def run_example(idx: int) -> tuple[str, float, gr.update, gr.update, gr.update, gr.update, gr.update]:
    """Run example."""
    img, thr, _, _ = EXAMPLES[int(idx)]

    img_local = download_image(img)
    result = run(img_local, thr)

    return (img_local, thr, *result)


with gr.Blocks(
    css="""
body {background:#020617;color:white}
img {max-height:400px; object-fit:contain;}
#chart {margin-top:10px;}
table {margin-top:30px;}
.kpi {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px;
    text-align: center;
}
.kpi-value {
    font-size: 20px;
    font-weight: 600;
}
.kpi-label {
    font-size: 12px;
    opacity: 0.7;
}
.spinner {
    width:50px;
    height:50px;
    border:4px solid rgba(255,255,255,0.1);
    border-top:4px solid #3b82f6;
    border-radius:50%;
    animation:spin 1s linear infinite;
}
.section-title {
    font-size:14px;
    font-weight:600;
    opacity:0.8;
    margin-top:20px;
    margin-bottom:10px;
}
tr:hover {
    background: rgba(255,255,255,0.05);
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""
) as demo:
    gr.Markdown('# 🛰️ Aircraft Detection Dashboard')
    example_trigger = gr.Textbox(visible=False, elem_id='example_trigger')
    loader = gr.HTML("""
    <div id="loader" style="
        display:none;
        position:fixed;
        top:0; left:0;
        width:100%; height:100%;
        background:rgba(2,6,23,0.85);
        z-index:9999;
        align-items:center;
        justify-content:center;
        flex-direction:column;
    ">
        <div class="spinner"></div>
        <div style="margin-top:15px;font-size:14px;opacity:0.8">
            Running detection...
        </div>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type='filepath', label='Input Image')
            thr = gr.Slider(0, 1, value=0.3, label='Confidence')
            btn = gr.Button('Run')

        with gr.Column(scale=1):
            img_output = gr.Image(label='Detection Result', height=400)

        with gr.Column(scale=1):
            loader = gr.Markdown('⏳ Statistics will be here...', visible=False)
            kpi = gr.HTML(value="<div style='opacity:0.4'>No results yet</div>")
            chart = gr.HTML(value='')

    table = gr.HTML(value='')
    examples_table = gr.HTML(
        """
    <div style="margin-top:30px; font-size:24px; font-weight:600; opacity:0.8">
        Example inputs and model predictions
    </div>
    """
        + build_examples_table()
    )
    example_trigger.change(
        fn=run_example, inputs=example_trigger, outputs=[img_input, thr, img_output, kpi, chart, table, loader], api_name=False
    )
    btn.click(fn=lambda: gr.update(visible=True), outputs=loader, api_name=False).then(
        fn=run, inputs=[img_input, thr], outputs=[img_output, kpi, chart, table, loader], api_name=False
    )

demo.launch(show_api=False)