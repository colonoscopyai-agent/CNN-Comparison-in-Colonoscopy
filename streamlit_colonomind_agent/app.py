
import os, urllib.request
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # match training (Keras 2) so weights + ViT load
import json
import joblib
import numpy as np
import pandas as pd
import cv2
import pywt
import scipy.stats
import streamlit as st

from pathlib import Path
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, BatchNormalization,
    Dropout, Concatenate, Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB4

try:
    from tensorflow.keras.applications import ConvNeXtTiny
    HAS_CONVNEXT = True
except Exception:
    HAS_CONVNEXT = False

try:
    from transformers import TFViTModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ============================================================
# 0. Page config + paths
# ============================================================
st.set_page_config(
    page_title="ColonoMind Diagnostic Agent",
    page_icon="\U0001F9E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("\U0001F9E0 ColonoMind Diagnostic Agent")
st.caption(
    "Upload a colonoscopy image, pick a trained model, run the diagnosis, "
    "review the model's report and ask questions. "
    "Research/educational tool — not a clinical diagnosis."
)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
REPORT_PATH = BASE_DIR / "deployment_report.json"
PREPROCESS_PATH = BASE_DIR / "preprocess_artifacts.joblib"

# ============================================================
# Cloud asset download (Streamlit Cloud / Hugging Face Spaces)
# Big weight files download at startup if not already present. Set ASSET_BASE_URL
# (Streamlit secrets or env) to a host that serves them, e.g. a GitHub Release:
#   https://github.com/<owner>/<repo>/releases/download/<tag>
# If the files are already beside app.py (committed / in the HF Space), no
# download happens and ASSET_BASE_URL is not needed.
# ============================================================
def _asset_base_url():
    url = os.environ.get("ASSET_BASE_URL", "")
    try:
        url = st.secrets.get("ASSET_BASE_URL", url)
    except Exception:
        pass
    return url.rstrip("/")

def ensure_asset(rel_path):
    dest = BASE_DIR / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    base = _asset_base_url()
    if not base:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading {rel_path} ..."):
        urllib.request.urlretrieve(f"{base}/{rel_path}", dest)
    return dest

ensure_asset("deployment_report.json")
ensure_asset("preprocess_artifacts.joblib")

# Clinical meaning of the Mayo Endoscopic Subscore classes (grounding for Q&A)
MES_INFO = {
    "MES0": "Mayo Endoscopic Subscore 0 — normal or inactive disease "
            "(intact vascular pattern, no friability).",
    "MES1": "Mayo Endoscopic Subscore 1 — mild activity "
            "(erythema, decreased vascular pattern, mild friability).",
    "MES2": "Mayo Endoscopic Subscore 2 — moderate activity "
            "(marked erythema, absent vascular pattern, friability, erosions).",
    "MES3": "Mayo Endoscopic Subscore 3 — severe activity "
            "(spontaneous bleeding, ulceration).",
}

# ============================================================
# 1. Load deployment report + preprocessing artifacts
# ============================================================
@st.cache_data
def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_preprocess_artifacts():
    return joblib.load(PREPROCESS_PATH)

report = load_report()
artifacts = load_preprocess_artifacts()

CLASS_NAMES = artifacts["class_names"]
IMG_SIZE = tuple(artifacts["img_size"])
WAVELET = artifacts.get("wavelet", "db1")
scaler = artifacts["scaler"]
umap_reducer = artifacts["umap_reducer"]

MODEL_REGISTRY = report["model_registry"]
METRICS_SUMMARY = report["metrics_summary"]
MES_INFO = report.get("class_descriptions", MES_INFO)

# ============================================================
# 2. Feature extraction  (MUST match the training notebook exactly)
#    17 DWT features (16 stats + HH energy) + 3 GLCM = 20 features
# ============================================================
def extract_wavelet_stats(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)

    def stats(subband):
        subband_abs = np.abs(subband.flatten()) + 1e-6
        return [
            float(np.mean(subband)),
            float(np.std(subband)),
            float(np.var(subband)),
            float(scipy.stats.entropy(subband_abs)),
        ]

    dwt_features = stats(LL) + stats(LH) + stats(HL) + stats(HH)   # 16
    dwt_features.append(float(np.sum(np.square(HH))))              # + HH energy -> 17
    return dwt_features

def extract_glcm_features(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    return [
        float(np.mean(graycoprops(glcm, "contrast"))),
        float(np.mean(graycoprops(glcm, "homogeneity"))),
        float(np.mean(graycoprops(glcm, "dissimilarity"))),
    ]                                                             # 3

def extract_combined_features(image_rgb):
    feats = extract_wavelet_stats(image_rgb) + extract_glcm_features(image_rgb)
    assert len(feats) == 20, f"Expected 20 features, got {len(feats)}"
    return np.array(feats, dtype=np.float32)

def preprocess_uploaded_image(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    image_rgb_original = np.array(pil_img)

    image_resized = cv2.resize(image_rgb_original, IMG_SIZE).astype(np.uint8)

    image_input = (image_resized.astype(np.float32) / 255.0)[np.newaxis, ...]

    handcrafted = extract_combined_features(image_resized).reshape(1, -1)
    handcrafted_scaled = scaler.transform(handcrafted)
    umap_feat = umap_reducer.transform(handcrafted_scaled)

    return {
        "pil_image": pil_img,
        "image_input": image_input,
        "handcrafted_scaled": handcrafted_scaled,
        "umap_feat": umap_feat,
    }

# ============================================================
# 3. Model architecture  (identical to the training notebook)
# ============================================================
def _cnn_head(base_model, dropout_rate):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x

def create_resnet50_branch(input_shape, dropout_rate=0.5):
    inp = Input(shape=input_shape, name="image_input_cnn")
    base = ResNet50(weights=None, include_top=False, input_tensor=inp)
    return Model(inp, _cnn_head(base, dropout_rate), name="ResNet50_Branch")

def create_densenet121_branch(input_shape, dropout_rate=0.5):
    inp = Input(shape=input_shape, name="image_input_cnn")
    base = DenseNet121(weights=None, include_top=False, input_tensor=inp)
    return Model(inp, _cnn_head(base, dropout_rate), name="DenseNet121_Branch")

def create_efficientnetb4_branch(input_shape, dropout_rate=0.5):
    inp = Input(shape=input_shape, name="image_input_cnn")
    base = EfficientNetB4(weights=None, include_top=False, input_tensor=inp)
    return Model(inp, _cnn_head(base, dropout_rate), name="EfficientNetB4_Branch")

def create_convnexttiny_branch(input_shape, dropout_rate=0.5):
    if not HAS_CONVNEXT:
        raise ImportError("ConvNeXtTiny is not available in this Keras version.")
    inp = Input(shape=input_shape, name="image_input_cnn")
    base = ConvNeXtTiny(weights=None, include_top=False, input_tensor=inp)
    return Model(inp, _cnn_head(base, dropout_rate), name="ConvNeXtTiny_Branch")

def create_vitb16_branch(input_shape, dropout_rate=0.5):
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is not installed — needed for ViT-B-16.")
    inp = Input(shape=input_shape, name="image_input_vit")
    vit = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit.trainable = False
    # TFViTModel expects pixel_values as channels-first (batch, 3, 224, 224)
    pv = Lambda(lambda t: tf.transpose(t, [0, 3, 1, 2]))(inp)
    outputs = vit(pixel_values=pv)
    cls_token = Lambda(lambda t: t[:, 0, :])(outputs.last_hidden_state)
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(cls_token)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inp, x, name="ViT_B16_Branch")

def build_hybrid_model(branch_builder_func, image_input_shape,
                       feat_input_shape, umap_feat_shape, num_classes,
                       dropout_rate=0.5):
    image_input = Input(shape=image_input_shape, name="image_input")
    x_cnn = branch_builder_func(image_input_shape, dropout_rate)(image_input)
    x_cnn = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)

    feat_input = Input(shape=feat_input_shape, name="feat_input")
    x_feat = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(feat_input)
    x_feat = BatchNormalization()(x_feat)
    x_feat = Dropout(dropout_rate)(x_feat)

    umap_input = Input(shape=umap_feat_shape, name="umap_input")
    x_umap = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(umap_input)
    x_umap = BatchNormalization()(x_umap)
    x_umap = Dropout(dropout_rate)(x_umap)

    combined = Concatenate()([x_cnn, x_feat, x_umap])
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(combined)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation="softmax", name="hybrid_output")(x)

    return Model(inputs=[image_input, feat_input, umap_input], outputs=output)

BRANCH_BUILDERS = {
    "ResNet-50": create_resnet50_branch,
    "DenseNet-121": create_densenet121_branch,
    "EfficientNet-B4": create_efficientnetb4_branch,
    "ConvNeXt-Tiny": create_convnexttiny_branch,
    "ViT-B-16": create_vitb16_branch,
}

@st.cache_resource(show_spinner=False)
def load_selected_model(model_name):
    model = build_hybrid_model(
        branch_builder_func=BRANCH_BUILDERS[model_name],
        image_input_shape=(224, 224, 3),
        feat_input_shape=(20,),
        umap_feat_shape=(2,),
        num_classes=len(CLASS_NAMES),
        dropout_rate=0.5,
    )
    _wf = MODEL_REGISTRY[model_name]["weights_path"]
    ensure_asset(f"weights/{_wf}")           # download from ASSET_BASE_URL if missing
    weight_path = WEIGHTS_DIR / _wf
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    model.load_weights(str(weight_path))
    return model

# ============================================================
# 4. Sidebar  —  model (weight) dropdown
# ============================================================
st.sidebar.header("⚙️ Model selection")
selected_model_name = st.sidebar.selectbox(
    "Choose which trained weight to run",
    list(MODEL_REGISTRY.keys()),
)
st.sidebar.success(f"Active model: {selected_model_name}")

use_llm = st.sidebar.checkbox("Use Claude for Q&A (optional)", value=False)
api_key_input = ""
if use_llm:
    api_key_input = st.sidebar.text_input(
        "ANTHROPIC_API_KEY", type="password",
        help="Leave empty to use the ANTHROPIC_API_KEY environment variable.",
    )

# ============================================================
# PART 1  —  Upload image  (top)
# ============================================================
st.markdown("---")
st.header("1️⃣ Upload image")

uploaded_file = st.file_uploader(
    "Upload a colonoscopy image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

processed = None
if uploaded_file is not None:
    processed = preprocess_uploaded_image(uploaded_file)
    st.image(processed["pil_image"], caption="Uploaded image", width=380)
else:
    st.info("Upload an image to begin.")

# ============================================================
# PART 2  —  Chosen-model report (metrics)  (middle)
# ============================================================
st.markdown("---")
st.header("2️⃣ Model performance report")

if selected_model_name in METRICS_SUMMARY:
    m = METRICS_SUMMARY[selected_model_name]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision_macro']:.3f}")
    c3.metric("Recall/Sens.", f"{m['recall_macro']:.3f}")
    c4.metric("Specificity", f"{m['specificity_macro']:.3f}")
    c5.metric("F1-score", f"{m['f1_macro']:.3f}")
    c6.metric("QWK", f"{m['qwk']:.3f}")

    cm = np.array(m["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {c}" for c in CLASS_NAMES],
        columns=[f"Pred {c}" for c in CLASS_NAMES],
    )
    st.markdown("**Confusion matrix (validation set)**")
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        "Metrics reflect this model's evaluation in the training notebook."
    )
else:
    st.warning("No metrics found for this model in deployment_report.json.")

# ============================================================
# PART 3  —  Image + classification result
# ============================================================
st.markdown("---")
st.header("3️⃣ Run diagnosis & result")

if processed is None:
    st.info("Please upload an image first.")
elif st.button("\U0001F50D Run diagnosis", type="primary"):
    with st.spinner(f"Running {selected_model_name}..."):
        model = load_selected_model(selected_model_name)
        y_proba = model.predict(
            [processed["image_input"],
             processed["handcrafted_scaled"],
             processed["umap_feat"]],
            verbose=0,
        )[0]

    pred_idx = int(np.argmax(y_proba))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(y_proba[pred_idx])

    st.session_state["last_prediction"] = {
        "model": selected_model_name,
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: float(y_proba[i])
                          for i in range(len(CLASS_NAMES))},
    }

# Persist the result across chat interactions
if "last_prediction" in st.session_state:
    pred = st.session_state["last_prediction"]
    left, right = st.columns([1, 1])
    with left:
        if processed is not None:
            st.image(processed["pil_image"], caption="Diagnosed image", width=340)
    with right:
        st.subheader("Classification result")
        st.success(f"Predicted class: **{pred['predicted_class']}**")
        st.metric("Confidence", f"{pred['confidence']:.3f}")
        st.caption(MES_INFO.get(pred["predicted_class"], ""))
        prob_df = pd.DataFrame({
            "Class": list(pred["probabilities"].keys()),
            "Probability": list(pred["probabilities"].values()),
        })
        st.bar_chart(prob_df.set_index("Class"))
        st.caption(f"Prediction produced by: {pred['model']}")

# ============================================================
# PART 4  —  Question & Answer agent  (bottom)
# ============================================================
st.markdown("---")
st.header("4️⃣ Diagnostic Q&A agent")
st.caption(
    "Ask about the selected model's metrics, the MES classes, or the last "
    "prediction. Answers are grounded in the deployment report."
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def build_context():
    return {
        "selected_model": selected_model_name,
        "class_names": CLASS_NAMES,
        "class_descriptions": MES_INFO,
        "selected_model_metrics": METRICS_SUMMARY.get(selected_model_name, {}),
        "last_prediction": st.session_state.get("last_prediction"),
    }

def deterministic_answer(query, ctx):
    q = query.lower()
    m = ctx["selected_model_metrics"]
    pred = ctx["last_prediction"]
    name = ctx["selected_model"]

    metric_keys = [
        ("accuracy", "accuracy", "accuracy"),
        ("precision", "precision_macro", "macro precision"),
        ("specificity", "specificity_macro", "macro specificity"),
        ("f1", "f1_macro", "macro F1-score"),
        ("qwk", "qwk", "quadratic weighted kappa"),
    ]
    for kw, key, label in metric_keys:
        if kw in q:
            val = m.get(key)
            return (f"**{name}** — {label}: "
                    f"{val:.4f}" if isinstance(val, (int, float))
                    else f"{label} is not available for {name}.")
    if "recall" in q or "sensitivity" in q:
        val = m.get("recall_macro")
        return f"**{name}** — macro recall/sensitivity: {val:.4f}" \
               if isinstance(val, (int, float)) else "Recall not available."
    if "confusion" in q:
        return f"Confusion matrix for **{name}**: {m.get('confusion_matrix')}"
    if any(k in q for k in ["mes", "class", "meaning", "severity", "score"]):
        return "\n\n".join(f"- **{k}**: {v}" for k, v in ctx["class_descriptions"].items())
    if any(k in q for k in ["prediction", "result", "diagnos", "predict"]):
        if pred is None:
            return "No prediction yet — upload an image and click **Run diagnosis**."
        desc = ctx["class_descriptions"].get(pred["predicted_class"], "")
        return (f"Latest prediction ({pred['model']}): **{pred['predicted_class']}** "
                f"at {pred['confidence']:.3f} confidence.\n\n{desc}\n\n"
                "This is model output, not a clinical diagnosis.")
    return ("I can answer about this model's accuracy, precision, recall, "
            "specificity, F1, QWK, confusion matrix, the MES classes, and the "
            "latest prediction.")

def claude_answer(query, ctx, api_key):
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            system=(
                "You are a diagnostic assistant for the ColonoMind app. "
                "Answer ONLY from the provided JSON context. Do not invent "
                "metrics or make clinical decisions. State that outputs are "
                "model classifications, not a final diagnosis. Be concise."
            ),
            messages=[{
                "role": "user",
                "content": f"Context JSON:\n{json.dumps(ctx, indent=2)}\n\nQuestion: {query}",
            }],
        )
        return msg.content[0].text
    except Exception as e:
        return deterministic_answer(query, ctx) + f"\n\n_(Claude unavailable: {e})_"

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask about this model or the prediction...")
if user_question:
    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    ctx = build_context()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if use_llm:
                answer = claude_answer(user_question, ctx, api_key_input)
            else:
                answer = deterministic_answer(user_question, ctx)
        st.markdown(answer)
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
