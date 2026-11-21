from pathlib import Path
import logging
import random
import time
import mlflow
import torch
import numpy as np 
import torch.nn as nn
import segmentation_models_pytorch as smp

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.Axion.dataset import SliceDataset
from src.Axion.unet_arch import UNet, count_parameters
from src.Axion.utils.metrics import dice_coefficient, iou_score

logger=logging.getLogger(__name__)

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=True

def flat(d,p="",sep="."):
    result = {}
    for key, value in d.items():
        name = f"{p}{sep}{key}" if p else key
        if isinstance(value, dict):
            result.update(flat(value, name, sep))
        else:
            result[name] = value
    return result


dice=smp.losses.DiceLoss(mode="binary",from_logits=True)
bce=nn.BCEWithLogitsLoss()

def criterion(pred,target): 
    return dice(pred,target)+bce(pred,target)


def run_epoch(model,
              loader,
              device,
              use_amp,
              optimizer=None,
              scaler=None,
              acc_steps=1):
    
    train=optimizer is not None
    model.train() if train else model.eval()

    total={"loss": 0.0,
           "dice": 0.0,
           "iou": 0.0
           }

    if train: optimizer.zero_grad()

    for i,batch in enumerate(tqdm(loader,leave=True)):
        x=batch["image"].to(device,non_blocking=True)
        y=batch["mask"].to(device,non_blocking=True)

        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)

        if train:
            group_size = min(acc_steps, len(loader) - (i // acc_steps) * acc_steps)
            scaler.scale(loss / group_size).backward() # pyright: ignore[reportOptionalMemberAccess]

            if (i+1)%acc_steps==0 or (i+1)==len(loader):
                scaler.step(optimizer) # pyright: ignore[reportOptionalMemberAccess]
                scaler.update() # pyright: ignore[reportOptionalMemberAccess]
                optimizer.zero_grad()

        total["loss"]+=loss.item()
        with torch.no_grad():
            total["dice"]+=dice_coefficient(out,y).item()
            total["iou"]+=iou_score(out,y).item()

    n=len(loader)
    return {k: v/n for k,v in total.items()}


def train(cfg, scans):
    tr, pp, mp = cfg["training"], cfg["preprocessing"], cfg["model"]
    set_seed(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_path = Path(cfg["inference"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda" and tr["mixed_precision"]
    pin_memory = device.type == "cuda" and tr["pin_memory"]
    if tr["epochs"] < 1 or tr["gradient_accumulation_steps"] < 1:
        raise ValueError("Training needs at least one epoch and a positive gradient accumulation interval")

    scans = list(scans)
    patients = sorted({scan.patient_id for scan in scans})
    if len(patients) < 2:
        raise ValueError("Training needs labeled DICOM slices from at least two distinct patients")
    if any(scan.masks is None for scan in scans):
        raise ValueError("Training requires a mask for every DICOM slice")

    train_patients, val_patients = train_test_split(patients, test_size=pp["val_split"], random_state=tr["seed"])
    train_ds = SliceDataset((scan for scan in scans if scan.patient_id in train_patients), cfg, augment=True)
    val_ds = SliceDataset((scan for scan in scans if scan.patient_id in val_patients), cfg)
    logger.info(
        "Training on %d patients (%d slices), validating on %d patients (%d slices)",
        len(train_patients), len(train_ds), len(val_patients), len(val_ds),
    )
    train_loader = DataLoader(train_ds, tr["batch_size"], True, num_workers=tr["num_workers"], pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, tr["batch_size"], False, num_workers=tr["num_workers"], pin_memory=pin_memory)

    model=UNet(mp["in_channels"],mp["out_channels"],mp["features"]).to(device)
    logger.info(f"Params: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 1e-5),
    )
    schedule = tr.get("lr_scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=schedule.get("factor", 0.5), patience=schedule.get("patience", 5),
    )
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp) # pyright: ignore[reportAttributeAccessIssue]

    mlflow.set_tracking_uri(cfg.get("mlflow",{}).get("tracking_uri","./mlruns"))
    mlflow.set_experiment(cfg.get("mlflow",{}).get("experiment_name","brats"))

    best_dice = float("-inf")
    
    with mlflow.start_run():
        mlflow.log_params(flat(cfg))

        for epoch in range(1,tr["epochs"]+1):
            t0=time.time()

            trn=run_epoch(model,train_loader,device,use_amp,optimizer,scaler,tr["gradient_accumulation_steps"])
            val=run_epoch(model,val_loader,device,use_amp)

            scheduler.step(val["dice"])
            print(
                f"E{epoch:03d} | "
                f"T-Dice {trn['dice']:.4f} | V-Dice {val['dice']:.4f} | V-IoU {val['iou']:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.5f} | "
                f"{time.time()-t0:.1f}s")
                
            mlflow.log_metrics({
                "train_dice": trn["dice"],
                "val_dice": val["dice"],
                "val_iou": val["iou"],
                "train_loss": trn["loss"],
                "val_loss": val["loss"],
                "lr": optimizer.param_groups[0]["lr"],},step=epoch)

            if val["dice"]>best_dice:
                best_dice=val["dice"]
                checkpoint={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,}
                
                torch.save(checkpoint, checkpoint_path)

        mlflow.log_metric("best_val_dice",best_dice)
        logger.info(f"Best Dice: {best_dice:.4f}")
    logger.info("Checkpoint saved to %s", checkpoint_path)
