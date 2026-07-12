# Face Recognition — Age, Gender & Ethnicity Detection

A Dash web app using three PyTorch models (ResNet50, MobileNetV2, EfficientNet-B0) to predict age, gender, and ethnicity from uploaded images or live webcam feed.

## Project Structure

```
.
├── app.py
├── predict.py
├── training.ipynb
├── requirements.txt
├── haarcascade_frontalface_default.xml
├── resnet50_age_gender_ethnicity.pth
├── mobilenetv2_age_gender_ethnicity.pth
└── efficientnet_age_gender_ethnicity.pth
```

All `.pth` files and the `.xml` must be in the same folder as `app.py`.

## Requirements

- Python 3.10–3.12
- Webcam (for Live Camera tab only)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> PyTorch installs the CPU build by default. For CUDA, install it manually from [pytorch.org](https://pytorch.org/get-started/locally/) before running the line above.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:8050` in your browser. Select a model, then choose **Image** to upload a photo or **Live Camera** to use your webcam.

## CLI Prediction (`predict.py`)

Runs a standalone ResNet50 age/gender prediction on a single image.

- Default weights file: `age_gender_model.pth`
- Default test image: `27.jpg`

Edit `model_path` or `test_image_path` in the script if needed, then:

```bash
python predict.py
```

## Training (`training.ipynb`)

Expected dataset structure:

```
./dataset/gender/Training/female/
./dataset/gender/Training/male/
./dataset/age/Training/<age-range>/
```

In Stage 6, uncomment the architecture you want (ResNet50, MobileNetV2, or EfficientNet) and run all cells. The `.pth` file will be saved to the project folder.

## Troubleshooting

- **`Warning: 'xxx.pth' not found`** — move the weights file into the same folder as `app.py`
- **Camera not working** — check that no other app is using the webcam and the browser has camera permission
- **Slow inference** — you're on CPU; install the CUDA build of PyTorch if you have an NVIDIA GPU