"""Gradio web interface for CT/MRI brain tumor detection.

Launch::

    python -m idh_glioma.app
    # or
    uv run tumor-app

Opens a browser-based UI for uploading CT/MRI images and getting
real-time tumor detection predictions with GradCAM heatmap overlays.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from idh_glioma.app_idh import predict_idh
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary

# ── Constants ────────────────────────────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_SIZE = 224
_CKPT_PATH = Path("checkpoints/mobilenetv3_ct_best.pt")

_DATASET_ROOT = Path("datasets/Kaggle_multimodal/Dataset")
_CT_TUMOR = _DATASET_ROOT / "Brain Tumor CT scan Images" / "Tumor"
_CT_HEALTHY = _DATASET_ROOT / "Brain Tumor CT scan Images" / "Healthy"
_MRI_TUMOR = _DATASET_ROOT / "Brain Tumor MRI images" / "Tumor"
_MRI_HEALTHY = _DATASET_ROOT / "Brain Tumor MRI images" / "Healthy"

_DEVICE: torch.device | None = None
_MODEL: torch.nn.Module | None = None

# Pre-compute normalization tensors (avoids repeated allocation)
_MEAN_TENSOR = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_STD_TENSOR = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

# ── Model loading ────────────────────────────────────────────────────
def _get_model() -> tuple[torch.nn.Module, torch.device]:
    global _MODEL, _DEVICE, _FWD_HOOK, _BWD_HOOK
    if _MODEL is not None and _DEVICE is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            _DEVICE = torch.device("cuda")
        except RuntimeError:
            pass

    _MODEL = build_mobilenetv3_binary(num_input_channels=3).to(_DEVICE)
    state = torch.load(_CKPT_PATH, map_location=_DEVICE, weights_only=True)
    _MODEL.load_state_dict(state["model"])
    _MODEL.eval()

    # Set inference-optimized mode
    torch.set_grad_enabled(False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    return _MODEL, _DEVICE


# ── GradCAM ──────────────────────────────────────────────────────────
def _gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute GradCAM heatmap from last convolutional layer."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.features[-1]

    def fwd_hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        activations.append(out.detach())

    def bwd_hook(_mod: torch.nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        gradients.append(grad_out[0].detach())

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    try:
        with torch.enable_grad():
            x = input_tensor.unsqueeze(0).to(device).requires_grad_(True)
            logit = model(x)
            model.zero_grad()
            logit.backward()

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(_IMG_SIZE, _IMG_SIZE), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam *= 1.0 / cam_max  # In-place normalize
    finally:
        fwd_h.remove()
        bwd_h.remove()

    return cam


def _overlay_heatmap(
    original: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay GradCAM heatmap on original image."""
    h, w = original.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * original)
    return overlay


# ── Prediction ───────────────────────────────────────────────────────
def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def predict(image: np.ndarray | None) -> tuple[dict[str, float], np.ndarray | None, str]:
    """Run tumor detection on uploaded image.

    Returns (label_confidences, gradcam_overlay, diagnosis_text).
    """
    if image is None:
        return {}, None, "Please upload a CT or MRI brain image."

    t0 = time.perf_counter()
    model, device = _get_model()

    pil_img = Image.fromarray(image).convert("RGB")
    tensor = _PREPROCESS(pil_img)

    # Inference (grad is globally disabled; GradCAM re-enables locally)
    logit = model(tensor.unsqueeze(0).to(device))
    prob_tumor = torch.sigmoid(logit).item()

    prob_healthy = 1.0 - prob_tumor
    is_tumor = prob_tumor >= 0.5

    # GradCAM (requires gradients — separate pass)
    cam = _gradcam(model, tensor, device)
    overlay = _overlay_heatmap(np.array(pil_img), cam)

    t_infer = time.perf_counter() - t0

    # ── Create comparison figure ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=100)

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM Heatmap", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    result_text = "Tumor" if is_tumor else "Healthy"
    confidence = prob_tumor if is_tumor else prob_healthy
    color = "#e74c3c" if is_tumor else "#27ae60"
    fig.suptitle(
        f"Result: {result_text}  |  Confidence: {confidence:.1%}  |  {t_infer:.0f}ms",
        fontsize=15,
        fontweight="bold",
        color=color,
        y=1.01,
    )
    fig.tight_layout()
    result_img = _fig_to_array(fig)

    # Diagnosis text
    diagnosis = _build_diagnosis(is_tumor, prob_tumor, t_infer)

    confidences = {"Tumor": prob_tumor, "Healthy": prob_healthy}
    return confidences, result_img, diagnosis


def _build_diagnosis(is_tumor: bool, prob: float, latency: float) -> str:
    if is_tumor:
        if prob >= 0.95:
            level = "High suspicion of tumor"
            detail = "The model detected tumor features with very high confidence. Further imaging (e.g., contrast-enhanced MRI) is recommended."
        elif prob >= 0.8:
            level = "Suspected tumor"
            detail = "The model detected significant tumor features. Specialist consultation is recommended."
        else:
            level = "Possible tumor"
            detail = "The model detected some tumor features with lower confidence. Follow-up or additional testing is recommended."
    else:
        if prob < 0.05:
            level = "Normal"
            detail = "No tumor features detected."
        elif prob < 0.2:
            level = "Likely normal"
            detail = "No obvious tumor features detected. Routine follow-up is recommended."
        else:
            level = "Borderline"
            detail = "The result is near the decision threshold. Further imaging is recommended for confirmation."

    return (
        f"## Analysis Result\n\n"
        f"**Assessment**: {level}\n\n"
        f"**Tumor probability**: {prob:.1%}\n\n"
        f"**Recommendation**: {detail}\n\n"
        f"**Inference time**: {latency * 1000:.0f} ms\n\n"
        f"---\n"
        f"*This result is for reference only and does not constitute a medical diagnosis. Please consult a physician.*"
    )


# ── Sample images ────────────────────────────────────────────────────
def _collect_examples() -> list[str]:
    """Gather a handful of example images from each category."""
    examples: list[str] = []
    for folder in [_CT_TUMOR, _CT_HEALTHY, _MRI_TUMOR, _MRI_HEALTHY]:
        if folder.is_dir():
            imgs = sorted(folder.glob("*.jpg"))[:2]
            examples.extend(str(p) for p in imgs)
    return examples


# ── Gradio UI ────────────────────────────────────────────────────────
def build_app():  # noqa: ANN201
    import os

    import gradio as gr

    # Use project-local cache to avoid /tmp permission issues
    cache_dir = Path("outputs/.gradio_cache").resolve()
    os.environ.setdefault("GRADIO_TEMP_DIR", str(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    examples = _collect_examples()

    # Pre-warm the model so first request is fast
    _get_model()

    with gr.Blocks() as app:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1.5em;">
                <h1>Brain Tumor Detection System</h1>
                <p style="color: #666;">
                    CT / MRI tumor detection &nbsp;+&nbsp; IDH mutation classification (TCGA-LGG)
                </p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("CT/MRI Tumor Detection"):
                gr.Markdown(
                    "Upload a single CT or MRI brain image (PNG/JPG). "
                    "Returns tumor probability + GradCAM heatmap. "
                    "Accuracy 96.4%, AUC 0.993 (Kaggle CT/MRI test split, 1,443 images)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload CT/MRI Image",
                            type="numpy",
                            height=350,
                        )
                        detect_btn = gr.Button("Detect", variant="primary", size="lg")
                        label_output = gr.Label(label="Classification", num_top_classes=2)

                    with gr.Column(scale=2):
                        result_image = gr.Image(
                            label="GradCAM Visualization",
                            type="numpy",
                            height=350,
                        )
                        diagnosis_md = gr.Markdown(
                            label="Report",
                            value="Upload an image and click **Detect** to begin analysis.",
                        )

                detect_btn.click(
                    fn=predict,
                    inputs=[input_image],
                    outputs=[label_output, result_image, diagnosis_md],
                )
                input_image.change(
                    fn=predict,
                    inputs=[input_image],
                    outputs=[label_output, result_image, diagnosis_md],
                )

                if examples:
                    gr.Examples(
                        examples=[[e] for e in examples],
                        inputs=[input_image],
                        outputs=[label_output, result_image, diagnosis_md],
                        fn=predict,
                        cache_examples=False,
                        label="Example Images (click to load)",
                    )

                with gr.Accordion("Model Info", open=False):
                    gr.Markdown(
                        """
                        | Item | Details |
                        |------|---------|
                        | **Architecture** | MobileNetV3-Small (binary classification) |
                        | **Training data** | 9,618 CT/MRI brain images |
                        | **Test accuracy** | 96.4% (1,443 test images) |
                        | **AUC-ROC** | 0.9928 |
                        | **Input size** | 224 x 224 RGB |

                        **GradCAM**: Red regions indicate the highest tumor correlation.
                        """
                    )

            with gr.Tab("IDH Mutation Classification (TCGA-LGG)"):
                gr.Markdown(
                    "Upload all four BraTS-style NIfTI volumes (`.nii.gz`) for one case. "
                    "The pipeline runs U-Net 2D segmentation, ROI-crops every tumor-bearing slice, "
                    "and averages MobileNetV3-large slice probabilities for a case-level IDH call. "
                    "Threshold (default 0.876) is the Youden's-J optimum on the val split."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        flair_in = gr.File(label="FLAIR (.nii.gz)", file_types=[".gz", ".nii"])
                        t1_in = gr.File(label="T1 (.nii.gz)", file_types=[".gz", ".nii"])
                        t1gd_in = gr.File(label="T1Gd (.nii.gz)", file_types=[".gz", ".nii"])
                        t2_in = gr.File(label="T2 (.nii.gz)", file_types=[".gz", ".nii"])
                        idh_btn = gr.Button("Run IDH analysis", variant="primary", size="lg")
                        idh_label_out = gr.Label(
                            label="IDH classification", num_top_classes=2
                        )
                    with gr.Column(scale=2):
                        idh_image_out = gr.Image(
                            label="Tumor segmentation overview",
                            type="numpy",
                            height=350,
                        )
                        idh_diag_md = gr.Markdown(
                            label="Report",
                            value="Upload all four modalities and click **Run IDH analysis**.",
                        )

                idh_btn.click(
                    fn=predict_idh,
                    inputs=[flair_in, t1_in, t1gd_in, t2_in],
                    outputs=[idh_label_out, idh_image_out, idh_diag_md],
                )

                with gr.Accordion("Model Info", open=False):
                    gr.Markdown(
                        """
                        | Item | Details |
                        |------|---------|
                        | **Pipeline** | U-Net 2D (segmentation) → ROI crop → MobileNetV3-large (classification) |
                        | **Cohort** | TCGA-LGG (45 train / 10 val / 10 test cases) |
                        | **Test case-level AUC** | 0.875 |
                        | **Test slice-level AUC** | 0.453 (per-slice noisy by design; case vote wins) |
                        | **Decision threshold** | 0.876 (Youden's J on val) |
                        | **WT recall (calibrated)** | 50% (vs 0% at threshold 0.5) |

                        Inputs must be 4-modality BraTS-style NIfTI from a single case. The four volumes must share the same shape.
                        """
                    )

    return app


def main() -> None:
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        app_kwargs={"title": "Brain Tumor Detection System"},
    )


if __name__ == "__main__":
    main()
