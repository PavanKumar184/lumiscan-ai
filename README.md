# LumiScan AI

LumiScan AI is a Flask and PyTorch chest X-ray analysis web app. It lets users upload a chest radiograph, run model inference, view the predicted condition, inspect a Grad-CAM heatmap, and read supporting clinical/context information.

## Features

- Chest X-ray upload through a web interface
- DR-Ultra image branch using a DenseNet121 visual backbone
- Disease prediction with confidence score
- Grad-CAM heatmap visualization
- Disease explanation text
- Dataset class distribution context
- Responsive clinical workstation-style UI

## Prerequisites

Install these before running the project:

- Python 3.10 or 3.11
- For deployment, Python 3.11.9 is recommended
- Git
- pip
- Optional: CUDA GPU drivers for faster inference

## Clone The Project

```bash
git clone https://github.com/PavanKumar184/lumiscan-ai.git
cd lumiscan-ai
```

## Create A Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

Install the project requirements:

```bash
pip install -r requirements.txt
```

Install the additional packages used by the Flask/PyTorch app:

```bash
pip install torch torchvision psutil flask-cors
```

If installing PyTorch fails, use the official PyTorch install selector:

https://pytorch.org/get-started/locally

For CPU-only installation, this usually works:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Required Model Files

Make sure these files exist in the project root:

```text
best_dr_ultra_densenet121_nih.pth
classes.json
class_weights.csv
```

The app will not start without:

```text
best_dr_ultra_densenet121_nih.pth
```

## Run The App

Recommended:

```bash
python run_app.py
```

Or run Flask directly:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How To Use

1. Open the local app URL in a browser.
2. Upload a chest X-ray image in PNG, JPG, or JPEG format.
3. Click `Start Analysis`.
4. Review the generated report:
   - predicted condition
   - confidence score
   - Grad-CAM heatmap
   - clinical explanation
   - dataset class distribution

## Troubleshooting

If port `5000` is already busy, change this line in `app.py`:

```python
app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
```

to another port, for example:

```python
app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, use_reloader=False)
```

Then open:

```text
http://127.0.0.1:8000
```

If the app is slow on first startup, that is normal. The PyTorch checkpoint loads before Flask becomes available.

If `torch` or `torchvision` errors appear, reinstall them using the official PyTorch command for your operating system and hardware.

## Medical Disclaimer

This project is for educational and research demonstration purposes. The AI output should not be treated as a final medical diagnosis. Always consult qualified medical professionals for clinical interpretation.
