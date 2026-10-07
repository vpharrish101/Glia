"""
Inference script: load trained UNet,run on processed 2.5D slices,
save predicted masks and overlay images.
"""

import os
import logging
import cv2
import yaml
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from src.unet_arch import UNet
from src.ETL.dataclass import BraTS_Dataset,val_trf
from src.utils.metrics import dice_coefficient_numpy,iou_score_numpy
from src.utils.visualize import overlay_mask_on_image

logger=logging.getLogger(__name__)


def load_config(config_path: str)->dict:
    with open(config_path,"r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path:str,
               cfg: dict,
               device: torch.device)->torch.nn.Module:
    model=UNet(in_channels=cfg["model"]["in_channels"],
               out_channels=cfg["model"]["out_channels"],
               features=cfg["model"]["features"],).to(device)

    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Loaded model from epoch %s (best dice: %s)",checkpoint.get("epoch","?"),checkpoint.get("best_dice","?"))

    return model


def run_inference(config_path: str):
    cfg=load_config(config_path)

    logging.basicConfig(
        level=cfg.get("logging",{}).get("level","INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s",device)

    threshold=cfg["inference"]["threshold"]
    pred_dir=cfg["paths"]["predictions"]
    overlay_dir=cfg["paths"]["overlays"]
    image_size=cfg["preprocessing"]["image_size"]
    num_channels=cfg["model"]["in_channels"]
    os.makedirs(pred_dir,exist_ok=True)
    os.makedirs(overlay_dir,exist_ok=True)
    model=load_model(cfg["inference"]["checkpoint"],cfg,device)


    dataset=BraTS_Dataset(img_dir=cfg["paths"]["val_images"],
                          mask_dir=cfg["paths"]["val_masks"],
                          transform=val_trf(image_size,num_channels),)
    
    loader=DataLoader(dataset,
                      batch_size=cfg["inference"]["batch_size"],
                      shuffle=False,
                      num_workers=2,
                      pin_memory=True)

    logger.info(f"Running inference on {len(dataset)} images...")

    all_dice,all_iou=[],[]

    with torch.no_grad():
        for batch in tqdm(loader,desc="Inference"):
            images=batch["image"].to(device,non_blocking=True)
            masks=batch["mask"]
            filenames=batch["filename"]

            with torch.amp.autocast("cuda",enabled=True): #type:ignore
                outputs=model(images)

            preds=torch.sigmoid(outputs).cpu().numpy()
            preds_binary=(preds > threshold).astype(np.uint8)
            images_np=images.cpu().numpy()
            masks_np=masks.numpy()

            for i,fname in enumerate(filenames):
                pred_mask=preds_binary[i,0]
                gt_mask=(masks_np[i,0]>0.5).astype(np.uint8)

                all_dice.append(dice_coefficient_numpy(pred_mask,gt_mask))
                all_iou.append(iou_score_numpy(pred_mask,gt_mask))

                cv2.imwrite(os.path.join(pred_dir,fname.replace(".png","_pred.png")),(pred_mask * 255).astype(np.uint8),)

                center_ch=num_channels//2
                img_gray=(images_np[i,center_ch]*255).astype(np.uint8)
                overlay=overlay_mask_on_image(img_gray,pred_mask)
                cv2.imwrite(os.path.join(overlay_dir,fname.replace(".png","_overlay.png")),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR),)

    mean_dice=np.mean(all_dice)
    mean_iou=np.mean(all_iou)
    print(f"\n{'=' * 40}")
    print(f"Inference Results")
    print(f"{'=' * 40}")
    print(f"Samples:    {len(all_dice)}")
    print(f"Mean Dice:  {mean_dice:.4f}")
    print(f"Mean IoU:   {mean_iou:.4f}")
    print(f"Predictions saved to: {pred_dir}")
    print(f"Overlays saved to:    {overlay_dir}")

