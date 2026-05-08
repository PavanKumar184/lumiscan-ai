from flask import Flask, render_template, request, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import datetime
import os
import json
import gc
import psutil
import traceback
import torch.nn as nn
import cv2
import random
import pandas as pd

# ============================================================
# PERFORMANCE METRICS (RANDOMIZED HIGH-SCORE FOR STARTUP)
# ============================================================

def display_model_performance():
    print("\n[INFO] DR-Ultra performance metrics will be loaded from the trained checkpoint.\n")


# ============================================================
# PERFORMANCE METRICS AFTER EACH PREDICTION
# ============================================================

def print_prediction_metrics():
    print("\n------------------ TRAINED DR-ULTRA METRICS ------------------")
    metrics = checkpoint.get("val_metrics", {}) if "checkpoint" in globals() else {}
    if not metrics:
        print("Checkpoint metrics unavailable.")
    for metric, value in metrics.items():
        print(f"{metric:<30}: {value}")
    print("--------------------------------------------------------------\n")


# ============================================================
# Flask setup
# ============================================================
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, resources={r"/predict": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000"]}})
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
GRADCAM_FOLDER = os.path.join("static", "gradcam")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

is_render_runtime = bool(os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"))
enable_gradcam_env = os.environ.get("ENABLE_GRADCAM")
ENABLE_GRADCAM = (
    enable_gradcam_env.strip().lower() in {"1", "true", "yes", "on"}
    if enable_gradcam_env is not None
    else not is_render_runtime
)

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
print(f"[INFO] Using device: {device}")
print(f"[INFO] Grad-CAM enabled: {ENABLE_GRADCAM}")

# ============================================================
# Load trained DR-Ultra model
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DR_ULTRA_MODEL_PATH = os.path.join(BASE_DIR, "best_dr_ultra_densenet121_nih.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")
CLASS_WEIGHTS_PATH = os.path.join(BASE_DIR, "class_weights.csv")


class DRUltraImageBranch(nn.Module):
    """DR-Ultra image branch using DenseNet121 as the visual backbone."""

    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.classifier.in_features, num_classes),
        )

    def forward(self, images):
        return self.model(images)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if not os.path.exists(DR_ULTRA_MODEL_PATH):
    raise FileNotFoundError(f"Trained DR-Ultra checkpoint not found: {DR_ULTRA_MODEL_PATH}")

print("[INFO] Loading trained DR-Ultra checkpoint...")
checkpoint = load_checkpoint(DR_ULTRA_MODEL_PATH)

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        disease_labels = json.load(f)
else:
    disease_labels = checkpoint["classes"]

num_classes = len(disease_labels)
dr_ultra_model = DRUltraImageBranch(num_classes=num_classes).to(device)
dr_ultra_model.load_state_dict(checkpoint["model_state_dict"])
dr_ultra_model.eval()

prediction_threshold = float(os.environ.get("PREDICTION_THRESHOLD", 0.75))
print(f"[INFO] DR-Ultra framework loaded.")
print(f"[INFO] Visual backbone: {checkpoint.get('visual_backbone', 'densenet121')}")
print(f"[INFO] Model classes ({num_classes}): {disease_labels}")
print(f"[INFO] Best validation metrics: {checkpoint.get('val_metrics', {})}")

# ============================================================
# Disease explanations
# ============================================================
disease_explanations = {
    "Atelectasis": "Partial collapse of lung tissue, reducing air volume and oxygen exchange.",
    "Consolidation": "Solidification of lung tissue due to fluid accumulation or infection.",
    "Infiltration": "Diffuse opacity suggesting inflammation or early infection.",
    "Pneumothorax": "Air in pleural space causing lung collapse.",
    "Edema": "Fluid accumulation in lung tissue.",
    "Emphysema": "Destruction of alveoli causing poor oxygen exchange.",
    "Fibrosis": "Scarring and stiffening of lung tissue.",
    "Effusion": "Fluid between lung and chest wall.",
    "Pneumonia": "Infection causing inflammation and consolidation.",
    "Pleural Thickening": "Thickened pleural lining.",
    "Cardiomegaly": "Enlarged heart.",
    "Nodule": "Small opacity indicating possible tumor.",
    "Mass": "Large abnormal growth.",
    "Hernia": "Abdominal contents entering chest cavity.",
    "Pleural_Thickening": "Thickening of the pleural lining around the lungs.",
    "Lung Lesion": "Abnormal localized tissue.",
    "Fracture": "Bone fracture.",
    "Lung Opacity": "White shadow indicating abnormal density.",
    "Enlarged Cardiomediastinum": "Widened mediastinum."
}

# ============================================================
# Class Distribution from trained NIH split
# ============================================================
if os.path.exists(CLASS_WEIGHTS_PATH):
    weights_df = pd.read_csv(CLASS_WEIGHTS_PATH)
    class_distribution = {
        row["class_name"]: int(row["positive_count"])
        for _, row in weights_df.iterrows()
    }
else:
    class_distribution = {cls: 0 for cls in disease_labels}

total_samples = sum(class_distribution.values())
avg_count = total_samples / max(len(class_distribution), 1)

# ============================================================
# Preprocessing used during Kaggle training
# ============================================================
dr_ultra_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# Memory cleanup
# ============================================================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"[MEMORY] Current usage: {mem:.2f} MB")

# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(img_path, model, target_layer, target_class, save_path):
    handle_f = None
    handle_b = None
    try:
        img = Image.open(img_path).convert("RGB")
        input_tensor = dr_ultra_transform(img).unsqueeze(0).to(device)

        activations, gradients = {}, {}

        def forward_hook(module, inputs, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        model.zero_grad()
        logits = model(input_tensor)
        loss = logits[0, target_class]
        loss.backward()

        grads = gradients["value"]
        acts = activations["value"]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = np.uint8(0.5 * heatmap + 0.5 * img_cv)
        cv2.imwrite(save_path, overlay)

        print(f"[GRADCAM] Saved heatmap at: {save_path}")
        return True

    except Exception as e:
        print(f"[GRADCAM ERROR] {str(e)}")
        return False
    finally:
        if handle_f is not None:
            handle_f.remove()
        if handle_b is not None:
            handle_b.remove()

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            raise ValueError("No file uploaded.")

        safe_filename = secure_filename(file.filename)
        if not safe_filename:
            raise ValueError("Invalid file name.")

        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)
        print(f"[INFO] Image saved: {filepath}")

        with Image.open(filepath) as img:
            dr_ultra_input = dr_ultra_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = dr_ultra_model(dr_ultra_input)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        prob_dict = {label: float(prob) for label, prob in zip(disease_labels, probs)}
        top_label = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_label]

        prediction_text = top_label if top_prob >= prediction_threshold else "Normal / No abnormality detected"

        print(f"[INFO] Prediction: {prediction_text} ({top_prob:.3f})")
        print(f"[INFO] All DR-Ultra probabilities: {prob_dict}")

        # ⭐ NEW: PRINT METRICS AFTER PREDICTION
        print_prediction_metrics()

        gradcam_url = None
        if ENABLE_GRADCAM:
            gradcam_filename = f"gradcam_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)

            gradcam_created = generate_gradcam(
                filepath,
                dr_ultra_model,
                dr_ultra_model.model.features.denseblock4,
                target_class=np.argmax(probs),
                save_path=gradcam_path
            )
            if gradcam_created:
                gradcam_url = url_for("static", filename=f"gradcam/{gradcam_filename}")
        else:
            print("[GRADCAM] Skipped. Set ENABLE_GRADCAM=true to enable heatmaps.")

        explanation_text = disease_explanations.get(
            prediction_text,
            "No disease-specific explanation available. Please correlate with clinical findings."
        )
        class_count = class_distribution.get(prediction_text, 0)
        imbalance_status = "Major Class (Overrepresented)" if class_count >= avg_count else "Minor Class (Underrepresented)"

        imbalance_info = {
            "total_samples": total_samples,
            "average_per_class": round(avg_count, 2),
            "class_samples": class_count,
            "imbalance_status": imbalance_status
        }

        result = {
            "scan_id": f"LUMI-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "prediction_result": prediction_text,
            "confidence": round(float(top_prob), 3) if prediction_text != "Normal / No abnormality detected" else None,
            "explanation_text": explanation_text,
            "image_url": url_for("static", filename=f"uploads/{safe_filename}"),
            "gradcam_url": gradcam_url,
            "gradcam_available": gradcam_url is not None,
            "imbalance_info": imbalance_info
        }

        return render_template("result.html", result=result)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {traceback.format_exc()}")

        result = {
            "scan_id": "N/A",
            "prediction_result": "Analysis Failed",
            "confidence": None,
            "explanation_text": f"Error: {str(e)}",
            "image_url": None,
            "gradcam_url": None,
            "gradcam_available": False,
            "imbalance_info": {
                "total_samples": total_samples,
                "average_per_class": round(avg_count, 2),
                "class_samples": 0,
                "imbalance_status": "N/A"
            }
        }
        return render_template("result.html", result=result)

    finally:
        clear_memory()

@app.route("/classes")
def classes():
    return render_template("class.html")

# ============================================================
# Run Flask App
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)

