"""Tune end-to-end IDH settings on a manifest split using macro F1.

Runs the full predicted-mask pipeline:
1. MONAI bundle segmentation
2. Configurable mask postprocess
3. Multi-view ROI extraction
4. 3D DenseNet IDH classification
5. Threshold search for macro F1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import NormalizeIntensity
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm

from idh_glioma.infer.e2e_roi import (
    apply_mask_postprocess,
    merge_e2e_config,
    predict_multi_view_idh,
    save_e2e_config,
    select_best_threshold,
)
from idh_glioma.utils import load_json


def _parse_view_margin_set(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--bundle-ckpt",
        type=Path,
        default=Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt"),
    )
    p.add_argument(
        "--cls-ckpt",
        type=Path,
        default=Path("checkpoints/densenet3d_idh_jitter.pt"),
    )
    p.add_argument("--seg-roi-size", type=int, nargs=3, default=(240, 240, 160))
    p.add_argument("--seg-overlap", type=float, default=0.5)
    p.add_argument(
        "--view-margin-set",
        dest="view_margin_sets",
        action="append",
        default=None,
        help="Comma-separated extra ROI margins. Repeat for multiple candidates, e.g. --view-margin-set 0 --view-margin-set 0,2",
    )
    p.add_argument("--dilate-iters", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--aggregation", nargs="*", default=["mean", "median"])
    p.add_argument("--threshold-start", type=float, default=0.05)
    p.add_argument("--threshold-stop", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.005)
    p.add_argument("--output", type=Path, default=Path("artifacts/e2e_idh_config.json"))
    return p.parse_args()


def _load_models(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, tuple[int, int, int], dict]:
    seg = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)
    seg_state = torch.load(args.bundle_ckpt, map_location=device, weights_only=True)
    if isinstance(seg_state, dict) and "state_dict" in seg_state:
        seg_state = seg_state["state_dict"]
    seg.load_state_dict(seg_state)
    seg.train(False)

    cls_state = torch.load(args.cls_ckpt, map_location=device, weights_only=True)
    cls = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    cls.load_state_dict(cls_state["model"])
    cls.train(False)
    target_size = tuple(cls_state.get("target_size", (96, 96, 96)))
    if device.type == "cuda":
        cls = cls.to(memory_format=torch.channels_last_3d)
    return seg, cls, target_size, cls_state


def _predict_seg_mask(
    seg: torch.nn.Module,
    normalize: NormalizeIntensity,
    device: torch.device,
    use_amp: bool,
    roi_size: tuple[int, int, int],
    overlap: float,
    flair: np.ndarray,
    t1: np.ndarray,
    t1gd: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    img4 = np.stack([t1gd, t1, t2, flair], axis=0)
    img_t = normalize(torch.from_numpy(img4)).unsqueeze(0).to(device).float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        seg_logits = sliding_window_inference(
            img_t,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=seg,
            overlap=overlap,
        )
    probs = torch.sigmoid(seg_logits).cpu().numpy()[0]
    return (probs[1] > 0.5).astype(np.uint8)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    manifest = load_json(args.manifest)
    records = [r for r in manifest[args.split] if r.get("idh_label") is not None]
    if not records:
        raise ValueError(f"{args.split!r} split has no labelled cases.")

    seg, cls, target_size, cls_state = _load_models(args, device)
    base_cfg = merge_e2e_config(
        None,
        fallback_threshold=float(cls_state.get("threshold", 0.5)),
        fallback_base_margin=int(cls_state.get("margin", 4)),
    )
    view_margin_sets = args.view_margin_sets or ["0", "0,2", "0,4"]
    view_margin_candidates = [_parse_view_margin_set(text) for text in view_margin_sets]
    thresholds = np.arange(args.threshold_start, args.threshold_stop + 1e-9, args.threshold_step)

    volumes = []
    for r in tqdm(records, desc="load-cases"):
        flair = nib.load(r["modalities"]["flair"]).get_fdata().astype(np.float32)
        t1 = nib.load(r["modalities"]["t1"]).get_fdata().astype(np.float32)
        t1gd = nib.load(r["modalities"]["t1Gd"]).get_fdata().astype(np.float32)
        t2 = nib.load(r["modalities"]["t2"]).get_fdata().astype(np.float32)
        raw_mask = _predict_seg_mask(
            seg,
            normalize,
            device,
            use_amp,
            tuple(args.seg_roi_size),
            args.seg_overlap,
            flair,
            t1,
            t1gd,
            t2,
        )
        volumes.append(
            {
                "case_id": r.get("case_id", ""),
                "label": int(r["idh_label"]),
                "flair": flair,
                "t1": t1,
                "t1gd": t1gd,
                "t2": t2,
                "raw_mask": raw_mask,
            }
        )

    best: dict | None = None
    for dilate_iters in args.dilate_iters:
        for aggregation in args.aggregation:
            for view_margins in view_margin_candidates:
                cfg = dict(base_cfg)
                cfg.update(
                    {
                        "dilate_iters": int(dilate_iters),
                        "aggregation": aggregation,
                        "view_margins": list(view_margins),
                    }
                )
                labels: list[int] = []
                probs: list[float] = []
                for item in tqdm(volumes, desc=f"sweep d={dilate_iters} agg={aggregation} vm={view_margins}", leave=False):
                    pred_mask = apply_mask_postprocess(
                        item["raw_mask"],
                        keep_largest=bool(cfg["keep_largest"]),
                        dilate_iters=int(cfg["dilate_iters"]),
                    )
                    if not pred_mask.any():
                        prob = float("nan")
                    else:
                        prob, _, _ = predict_multi_view_idh(
                            flair=item["flair"],
                            t1=item["t1"],
                            t1gd=item["t1gd"],
                            t2=item["t2"],
                            pred_mask=pred_mask,
                            cls_model=cls,
                            device=device,
                            target_size=target_size,
                            cfg=cfg,
                            use_amp=use_amp,
                        )
                    labels.append(int(item["label"]))
                    probs.append(prob)

                labels_arr = np.asarray(labels, dtype=np.int64)
                probs_arr = np.asarray(probs, dtype=np.float64)
                valid = ~np.isnan(probs_arr)
                if valid.sum() == 0:
                    continue
                valid_labels = labels_arr[valid]
                valid_probs = probs_arr[valid]
                threshold, macro_f1 = select_best_threshold(
                    valid_labels,
                    valid_probs,
                    thresholds,
                    preferred_threshold=float(base_cfg["threshold"]),
                )
                preds = (valid_probs >= threshold).astype(np.int64)
                result = {
                    "threshold": float(threshold),
                    "macro_f1": float(macro_f1),
                    "accuracy": float(accuracy_score(valid_labels, preds)),
                    "wt_recall": float(recall_score(valid_labels, preds, pos_label=0, zero_division=0)),
                    "mutant_recall": float(recall_score(valid_labels, preds, pos_label=1, zero_division=0)),
                    "auc": float(roc_auc_score(valid_labels, valid_probs)) if len(np.unique(valid_labels)) >= 2 else float("nan"),
                    "keep_largest": bool(cfg["keep_largest"]),
                    "dilate_iters": int(cfg["dilate_iters"]),
                    "base_margin": int(cfg["base_margin"]),
                    "view_margins": list(cfg["view_margins"]),
                    "aggregation": str(cfg["aggregation"]),
                    "threshold_objective": "macro_f1",
                    "threshold_split": args.split,
                    "num_valid_cases": int(valid.sum()),
                    "num_total_cases": len(records),
                    "cls_ckpt": str(args.cls_ckpt),
                    "bundle_ckpt": str(args.bundle_ckpt),
                }
                if best is None or result["macro_f1"] > best["macro_f1"]:
                    best = result
                    print(
                        f"[best] macro_f1={best['macro_f1']:.4f} threshold={best['threshold']:.4f} "
                        f"agg={best['aggregation']} view_margins={best['view_margins']} "
                        f"dilate_iters={best['dilate_iters']} wt_recall={best['wt_recall']:.4f} "
                        f"mutant_recall={best['mutant_recall']:.4f} auc={best['auc']:.4f}"
                    )

    if best is None:
        raise RuntimeError("No valid end-to-end configuration produced any probabilities.")

    save_e2e_config(args.output, best)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
