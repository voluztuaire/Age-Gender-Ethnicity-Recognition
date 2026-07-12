import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import base64
import io

# --- 1. SETUP & MODEL DEFINITIONS ---
ETHNICITY_MAP = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
GENDER_MAP = {0: 'Male', 1: 'Female'}

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, in_features, num_ethnicities):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.age_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.gender_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
        self.ethnicity_head = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_ethnicities))
    def forward(self, x):
        features = self.base_model(x)
        return (self.age_head(features).squeeze(1), self.gender_head(features), self.ethnicity_head(features))

def get_model_architecture(model_name, num_ethnicities):
    if model_name == 'resnet50':
        base = models.resnet50(weights=None)
        in_features = base.fc.in_features; base.fc = nn.Identity()
    elif model_name == 'mobilenetv2':
        base = models.mobilenet_v2(weights=None)
        in_features = base.classifier[1].in_features; base.classifier = nn.Identity()
    elif model_name == 'efficientnet':
        base = models.efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features; base.classifier = nn.Identity()
    else: raise ValueError("Unknown model name")
    return MultiTaskModel(base, in_features, num_ethnicities)

# --- 2. LOAD MODELS ---
def load_all_models():
    print("Loading all available models...")
    device = torch.device("cpu")
    models_dict = {}
    model_names = ['resnet50', 'mobilenetv2', 'efficientnet']
    num_ethnicities = 5

    for name in model_names:
        try:
            model_path = f"{name}_age_gender_ethnicity.pth"
            model = get_model_architecture(name, num_ethnicities)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models_dict[name] = model
            print(f"-> Weights for '{name}' loaded successfully.")
        except FileNotFoundError:
            print(f"-> Warning: '{model_path}' not found. This model will not be available.")
        except Exception as e:
            print(f"-> Warning: Failed to load '{name}'. Error: {e}")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Model initialization complete.")
    return models_dict, face_cascade

models_dict, face_cascade = load_all_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. DASH APP INITIALIZATION ---
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
    ],
    title="Neural Vision // Face Analysis",
    suppress_callback_exceptions=True
)
server = app.server

# --- CSS ---
CUSTOM_CSS = """
:root {
    --c-black: #000000;
    --c-bg: #1F150C;
    --c-accent: #412D15;
    --c-fg: #E1DCC9;
    --c-fg-dim: rgba(225, 220, 201, 0.65);
    --c-fg-faint: rgba(225, 220, 201, 0.15);
    --c-glass: rgba(225, 220, 201, 0.04);
    --c-glass-border: rgba(225, 220, 201, 0.12);
}

* { box-sizing: border-box; }

html, body, #react-entry-point, .dash-app-content {
    background: var(--c-bg) !important;
    color: var(--c-fg) !important;
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
    margin: 0;
    letter-spacing: 0.01em;
}

/* Ambient background glow */
body::before {
    content: "";
    position: fixed; inset: 0;
    background:
        radial-gradient(circle at 15% 20%, rgba(65, 45, 21, 0.55), transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(65, 45, 21, 0.40), transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(0, 0, 0, 0.6), transparent 70%);
    z-index: -2;
    pointer-events: none;
}
body::after {
    content: "";
    position: fixed; inset: 0;
    background-image:
        linear-gradient(var(--c-fg-faint) 1px, transparent 1px),
        linear-gradient(90deg, var(--c-fg-faint) 1px, transparent 1px);
    background-size: 64px 64px;
    mask-image: radial-gradient(circle at center, black 30%, transparent 80%);
    opacity: 0.25;
    z-index: -1;
    pointer-events: none;
}

/* ---------- Top bar ---------- */
.nv-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 32px;
    margin-bottom: 28px;
    background: var(--c-glass);
    backdrop-filter: blur(18px);
    border: 1px solid var(--c-glass-border);
    border-radius: 18px;
}
.nv-brand {
    display: flex; align-items: center; gap: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--c-fg);
}
.nv-brand-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--c-fg);
    box-shadow: 0 0 12px var(--c-fg);
}
.nv-brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: none;
}
.nv-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--c-fg-dim);
    display: flex; align-items: center; gap: 10px;
}
.nv-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--c-fg);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ---------- Glass panels ---------- */
.nv-panel {
    background: var(--c-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--c-glass-border);
    border-radius: 20px;
    padding: 28px;
    position: relative;
    overflow: hidden;
}
.nv-panel::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: var(--c-fg-faint);
}

.nv-panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--c-fg-dim);
    margin-bottom: 6px;
}
.nv-panel-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--c-fg);
    margin-bottom: 22px;
    letter-spacing: -0.01em;
}

.nv-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--c-fg-dim);
    margin: 18px 0 10px;
}

/* ---------- Dropdown ---------- */
.Select-control, .Select-menu-outer, .Select-value,
.nv-dropdown .Select-control {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid var(--c-glass-border) !important;
    border-radius: 12px !important;
    color: var(--c-fg) !important;
    min-height: 46px !important;
}
.Select-value-label, .Select-placeholder, .Select-input > input {
    color: var(--c-fg) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    line-height: 44px !important;
}
.Select-menu-outer {
    background: #100a06 !important;
    border: 1px solid var(--c-glass-border) !important;
    margin-top: 6px;
    border-radius: 12px !important;
    overflow: hidden;
}
.Select-option { background: transparent !important; color: var(--c-fg) !important; padding: 10px 14px !important; }
.Select-option.is-focused { background: var(--c-accent) !important; }
.Select-arrow { border-color: var(--c-fg) transparent transparent !important; }

/* ---------- Tabs ---------- */
.nav-tabs { border: none !important; gap: 8px; margin-bottom: 8px; }
.nav-tabs .nav-link {
    background: transparent !important;
    border: 1px solid var(--c-glass-border) !important;
    color: var(--c-fg-dim) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.nav-tabs .nav-link:hover { color: var(--c-fg) !important; border-color: var(--c-fg-dim) !important; }
.nav-tabs .nav-link.active {
    background: var(--c-accent) !important;
    color: var(--c-fg) !important;
    border-color: var(--c-accent) !important;
}

/* ---------- Upload dropzone ---------- */
.nv-upload {
    border: 1px dashed var(--c-glass-border) !important;
    border-radius: 14px;
    background: rgba(0,0,0,0.25);
    padding: 38px 20px !important;
    text-align: center;
    color: var(--c-fg-dim);
    font-family: 'Space Grotesk', sans-serif;
    cursor: pointer;
    transition: all 0.2s ease;
}
.nv-upload:hover {
    border-color: var(--c-fg-dim) !important;
    background: rgba(0,0,0,0.4);
    color: var(--c-fg);
}
.nv-upload a { color: var(--c-fg); text-decoration: underline; }

/* ---------- Button (solid, no gradient) ---------- */
.nv-btn {
    width: 100%;
    background: var(--c-accent) !important;
    color: var(--c-fg) !important;
    border: 1px solid var(--c-glass-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 500 !important;
    transition: all 0.2s ease;
    box-shadow: none !important;
}
.nv-btn:hover {
    background: #523a1d !important;
    border-color: var(--c-fg-dim) !important;
}
.nv-btn:focus, .nv-btn:active { background: var(--c-accent) !important; box-shadow: none !important; outline: none !important; }

/* ---------- Output area ---------- */
.nv-output {
    min-height: 480px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--c-glass-border);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}
.nv-output img { max-width: 100%; height: auto; border-radius: 10px; display: block; }
.nv-placeholder {
    text-align: center;
    color: var(--c-fg-dim);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.nv-placeholder-icon {
    width: 64px; height: 64px;
    margin: 0 auto 18px;
    border: 1px solid var(--c-glass-border);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    color: var(--c-fg);
    font-size: 22px;
}

/* ---------- Spinner color ---------- */
.spinner-border { color: var(--c-fg) !important; }

/* ---------- Alerts ---------- */
.alert {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid var(--c-glass-border) !important;
    color: var(--c-fg) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
}

/* corner ticks for that techy feel */
.nv-corner {
    position: absolute; width: 14px; height: 14px;
    border-color: var(--c-fg-dim);
    border-style: solid;
    border-width: 0;
}
.nv-corner.tl { top: 10px; left: 10px; border-top-width: 1px; border-left-width: 1px; }
.nv-corner.tr { top: 10px; right: 10px; border-top-width: 1px; border-right-width: 1px; }
.nv-corner.bl { bottom: 10px; left: 10px; border-bottom-width: 1px; border-left-width: 1px; }
.nv-corner.br { bottom: 10px; right: 10px; border-bottom-width: 1px; border-right-width: 1px; }

/* ---------- Webcam (browser-side) ---------- */
#webcam-video {
    width: 100%;
    max-width: 640px;
    border-radius: 10px;
    display: block;
    margin: 0 auto;
}
#webcam-canvas { display: none; }
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CUSTOM_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </footer>
</body>
</html>
"""

# --- 4. APP LAYOUT ---
def placeholder_block(text):
    return html.Div([
        html.Div("◉", className="nv-placeholder-icon"),
        html.Div(text),
    ], className="nv-placeholder")

app.layout = html.Div(
    style={"padding": "28px", "maxWidth": "1400px", "margin": "0 auto"},
    children=[
        # Top bar
        html.Div(className="nv-topbar", children=[
            html.Div(className="nv-brand", children=[
                html.Div(className="nv-brand-dot"),
                html.Span("NEURAL//VISION"),
                html.Span(" · ", style={"opacity": 0.4}),
                html.Span("Face Analysis Engine", className="nv-brand-title"),
            ]),
            html.Div(className="nv-status", children=[
                html.Div(className="nv-status-dot"),
                html.Span(f"{len(models_dict)} models online"),
            ]),
        ]),

        dbc.Row([
            # ---------------- Controls ----------------
            dbc.Col(md=4, children=[
                html.Div(className="nv-panel", children=[
                    html.Span(className="nv-corner tl"), html.Span(className="nv-corner tr"),
                    html.Span(className="nv-corner bl"), html.Span(className="nv-corner br"),

                    html.Div("CONTROL // 01", className="nv-panel-label"),
                    html.Div("Configuration", className="nv-panel-title"),

                    html.Div("Prediction Model", className="nv-section-label"),
                    dcc.Dropdown(
                        id='model-selector-dropdown',
                        className="nv-dropdown",
                        options=[{'label': name.upper(), 'value': name} for name in models_dict.keys()],
                        value=next(iter(models_dict.keys()), None),
                        clearable=False,
                    ),

                    html.Div("Input Source", className="nv-section-label"),
                    dbc.Tabs(id="input-method-tabs", active_tab="tab-upload", children=[
                        dbc.Tab(label="Image", tab_id="tab-upload"),
                        dbc.Tab(label="Live Camera", tab_id="tab-camera"),
                    ]),
                    html.Div(id="tab-content", style={"marginTop": "18px"}),
                ]),
            ]),

            # ---------------- Output ----------------
            dbc.Col(md=8, children=[
                html.Div(className="nv-panel", children=[
                    html.Span(className="nv-corner tl"), html.Span(className="nv-corner tr"),
                    html.Span(className="nv-corner bl"), html.Span(className="nv-corner br"),

                    html.Div("OUTPUT // 02", className="nv-panel-label"),
                    html.Div("Detection Result", className="nv-panel-title"),

                    html.Div(className="nv-output", children=[
                        dbc.Spinner(html.Div(id='output-display', children=placeholder_block("awaiting input")),
                                    color="light"),
                    ]),
                ]),
            ]),
        ], className="g-4"),

        # Hidden store that receives base64 frames captured by the browser JS
        dcc.Store(id='webcam-frame-store'),
        # Interval that triggers periodic frame capture on the client
        dcc.Interval(id='webcam-interval', interval=1200, disabled=True),
    ],
)

# --- 5. CALLBACKS ---
@app.callback(Output('output-display', 'children'),
              Output('tab-content', 'children'),
              Output('webcam-interval', 'disabled'),
              Input('input-method-tabs', 'active_tab'))
def switch_tab(at):
    if at == "tab-upload":
        upload_content = dcc.Upload(
            id='upload-image',
            className="nv-upload",
            children=html.Div(['Drag and drop or ', html.A('select an image file')]),
        )
        return placeholder_block("upload an image to begin"), upload_content, True

    if at == "tab-camera":
        camera_content = html.Div([
            dbc.Button('Start / Stop Camera', id='start-camera-button', className="nv-btn", n_clicks=0),
        ])
        return placeholder_block("camera idle"), camera_content, True

    return "Something went wrong", html.Div(), True


@app.callback(
    Output('output-display', 'children', allow_duplicate=True),
    Output('webcam-interval', 'disabled', allow_duplicate=True),
    Input('start-camera-button', 'n_clicks'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def toggle_camera(n_clicks, model_name):
    if n_clicks % 2 == 1:
        if not model_name:
            return dbc.Alert("Please select a model first.", color="warning"), True

        # This raw HTML block runs the browser's own webcam via getUserMedia.
        # It captures a frame to a hidden canvas and stores the base64 JPEG
        # in a hidden <div id="webcam-frame-store-raw"> that Dash's clientside
        # callback reads and syncs into dcc.Store.
        webcam_html = """
        <div>
            <video id="webcam-video" autoplay playsinline muted></video>
            <canvas id="webcam-canvas"></canvas>
        </div>
        <script>
        (function() {
            const video = document.getElementById('webcam-video');
            const canvas = document.getElementById('webcam-canvas');
            if (!video || !canvas) return;

            if (window._webcamStream) {
                window._webcamStream.getTracks().forEach(t => t.stop());
            }

            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    window._webcamStream = stream;
                    video.srcObject = stream;
                })
                .catch(function(err) {
                    console.error('Webcam access denied or unavailable:', err);
                });
        })();
        </script>
        """
        return html.Iframe(
            srcDoc=webcam_html,
            style={"width": "100%", "height": "480px", "border": "none"},
            id="webcam-frame"
        ), False
    else:
        if 'window' in dir():
            pass
        return placeholder_block("camera idle"), True


# Clientside callback: every interval tick, grab a frame from the <video>
# inside the iframe, draw it to canvas, convert to base64 JPEG, and push
# it into dcc.Store so a normal Dash callback can process it server-side.
app.clientside_callback(
    """
    function(n_intervals) {
        const iframe = document.getElementById('webcam-frame');
        if (!iframe) { return window.dash_clientside.no_update; }
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        const video = doc.getElementById('webcam-video');
        const canvas = doc.getElementById('webcam-canvas');
        if (!video || !canvas || video.readyState < 2) {
            return window.dash_clientside.no_update;
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        return dataUrl;
    }
    """,
    Output('webcam-frame-store', 'data'),
    Input('webcam-interval', 'n_intervals'),
)


def run_inference_on_image(image_pil, model):
    """Shared inference routine for both uploaded images and webcam frames."""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (225, 220, 201), 2)
        face_roi = image_pil.crop((x, y, x + w, y + h))
        image_tensor = transform(face_roi).unsqueeze(0)
        with torch.no_grad():
            age_pred, gender_pred, ethnicity_pred = model(image_tensor)
            gender = GENDER_MAP[torch.sigmoid(gender_pred).round().long().item()]
            ethnicity = ETHNICITY_MAP[torch.max(ethnicity_pred, 1)[1].item()]
            age = f"{age_pred.item():.1f}"
        text = f"{ethnicity}, {gender}, {age} yrs"
        cv2.putText(image_cv, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 220, 201), 2)

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img_result = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img_result.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.callback(
    Output('output-display', 'children', allow_duplicate=True),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def update_upload_output(contents, filename, model_name):
    if contents is None:
        return placeholder_block("upload an image to begin")
    if not model_name:
        return dbc.Alert("Please select a model first.", color="warning")

    model = models_dict[model_name]
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image_pil = Image.open(io.BytesIO(decoded)).convert('RGB')

    encoded_image_string = run_inference_on_image(image_pil, model)
    return html.Img(src=f'data:image/png;base64,{encoded_image_string}')


@app.callback(
    Output('output-display', 'children', allow_duplicate=True),
    Input('webcam-frame-store', 'data'),
    State('model-selector-dropdown', 'value'),
    prevent_initial_call=True
)
def process_webcam_frame(data_url, model_name):
    if not data_url:
        return dash.no_update
    if not model_name:
        return dbc.Alert("Please select a model first.", color="warning")

    try:
        header, content_string = data_url.split(',')
        decoded = base64.b64decode(content_string)
        image_pil = Image.open(io.BytesIO(decoded)).convert('RGB')
    except Exception:
        return dash.no_update

    model = models_dict[model_name]
    encoded_image_string = run_inference_on_image(image_pil, model)
    return html.Img(src=f'data:image/png;base64,{encoded_image_string}')


# --- 6. RUN APP ---
if __name__ == '__main__':
    app.run(debug=True)