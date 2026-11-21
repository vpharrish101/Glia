"""Predict segmentation masks for prepared image stacks."""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Axion.dataset import SliceDataset
from src.Axion.unet_arch import UNet
from src.Axion.utils.visualize import overlay_mask_on_image

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    settings = cfg["model"]
    model = UNet(settings["in_channels"], settings["out_channels"], settings["features"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded model from epoch %s (best Dice: %s)",
        checkpoint.get("epoch", "?"), checkpoint.get("best_dice", "?"),
    )
    return model


def run_inference(cfg, scans):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["inference"]["checkpoint"], cfg, device)
    pred_dir = Path(cfg["paths"]["predictions"])
    overlay_dir = Path(cfg["paths"]["overlays"])
    center = cfg["model"]["in_channels"] // 2
    threshold = cfg["inference"]["threshold"]
    outputs = {}

    with torch.inference_mode():
        for scan in scans:
            pred_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            loader = DataLoader(
                SliceDataset([scan], cfg), batch_size=cfg["inference"]["batch_size"],
                pin_memory=device.type == "cuda",
            )
            for batch in tqdm(loader, desc="Inference"):
                images = batch["image"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                masks = (logits[:, 0].float().sigmoid().cpu().numpy() > threshold).astype(np.uint8)

                for index, slice_id in enumerate(batch["slice_id"]):
                    pred_path = pred_dir / f"{slice_id}_pred.png"
                    overlay_path = overlay_dir / f"{slice_id}_overlay.png"
                    pixels = (batch["image"][index, center].numpy() * 255).astype(np.uint8)
                    overlay = overlay_mask_on_image(pixels, masks[index])
                    if not cv2.imwrite(str(pred_path), masks[index] * 255):
                        raise OSError(f"Could not write prediction for {slice_id}")
                    if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Could not write overlay for {slice_id}")
                    outputs[slice_id] = (pred_path, overlay_path)
    return outputs
