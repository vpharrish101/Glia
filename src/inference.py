"""Run labeled validation or predict on unlabeled, sanitized DICOM series."""

import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ETL.dataclass import BraTS_Dataset, val_trf
from src.ETL.preprocessing import context_slices
from src.unet_arch import UNet
from src.utils.metrics import dice_coefficient_numpy, iou_score_numpy
from src.utils.visualize import overlay_mask_on_image

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as stream:
        return yaml.safe_load(stream)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    settings = cfg["model"]
    model = UNet(settings["in_channels"], settings["out_channels"], settings["features"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded model from epoch %s (best dice: %s)", checkpoint.get("epoch", "?"), checkpoint.get("best_dice", "?"))
    return model


def _predict(dataset, model, cfg, device, pred_dir, overlay_dir):
    if not len(dataset):
        raise ValueError("No images available for inference")
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=cfg["inference"]["batch_size"], pin_memory=device.type == "cuda"
    )
    center_channel = cfg["model"]["in_channels"] // 2
    threshold = cfg["inference"]["threshold"]
    scores = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            masks = (logits[:, 0].float().sigmoid().cpu().numpy() > threshold).astype(np.uint8)
            for index, filename in enumerate(batch["filename"]):
                stem = Path(filename).stem
                predicted = masks[index]
                if "mask" in batch:
                    target = batch["mask"][index, 0].numpy() > 0.5
                    scores.append((dice_coefficient_numpy(predicted, target), iou_score_numpy(predicted, target)))

                image = (batch["image"][index, center_channel].numpy() * 255).astype(np.uint8)
                overlay = overlay_mask_on_image(image, predicted)
                if not cv2.imwrite(str(pred_dir / f"{stem}_pred.png"), predicted * 255):
                    raise OSError(f"Could not write prediction for {stem}")
                if not cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write overlay for {stem}")
    return scores


def run_inference(config_path: str):
    cfg = load_config(config_path)
    context_slices(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = cfg["paths"]
    dataset = BraTS_Dataset(
        paths["val_images"],
        paths["val_masks"],
        val_trf(cfg["preprocessing"]["image_size"], cfg["model"]["in_channels"]),
        paths.get("metadata"),
        "val",
    )
    model = load_model(cfg["inference"]["checkpoint"], cfg, device)
    scores = _predict(dataset, model, cfg, device, Path(paths["predictions"]), Path(paths["overlays"]))
    dice, iou = np.mean(scores, axis=0)
    print(f"Samples: {len(scores)}\nMean Dice: {dice:.4f}\nMean IoU: {iou:.4f}")


def run_dicom_inference(cfg, model, device, lake, records):
    from src.ETL.dicom_dataset import DicomDataset

    series = defaultdict(list)
    for record in records:
        series[(record["patient_id"], record["study_uid"], record["series_uid"])].append(record)

    pred_dir = Path(cfg["paths"]["predictions"])
    overlay_dir = Path(cfg["paths"]["overlays"])
    context = context_slices(cfg)
    report = []
    for key, entries in sorted(series.items()):
        dataset = DicomDataset(lake, entries, cfg["preprocessing"]["image_size"], context)
        _predict(dataset, model, cfg, device, pred_dir, overlay_dir)
        by_uid = {record["sop_uid"]: record for record in entries}
        for index, uid in enumerate(dataset.sop_uids):
            report.append((key[1], key[2], uid, index, by_uid[uid]["storage_key"],
                           pred_dir / f"{uid}_pred.png", overlay_dir / f"{uid}_overlay.png"))

    manifest = pred_dir / "dicom_predictions.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("study_uid", "series_uid", "sop_uid", "slice_index", "storage_key", "prediction", "overlay"))
        writer.writerows(report)
    logger.info("Predicted %d DICOM slices from %d series. Manifest: %s", len(report), len(series), manifest)
