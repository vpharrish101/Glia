import os
import argparse
import yaml
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
from src.ETL.dataclass import BraTS_Dataset,train_trf,val_trf
from src.unet_arch import UNet,count_parameters
from src.utils.metrics import dice_coefficient,iou_score

logger=logging.getLogger(__name__)

def load_config(p):
    return yaml.safe_load(open(p))

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=True

def flat(d,p="",sep="."):
    return {f"{p}{sep}{k}" if p else k: v
            for k,val in d.items()
            for v in (flat(val,f"{p}{sep}{k}" if p else k).items()
                      if isinstance(val,dict) else [(k,val)])}


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

        with torch.amp.autocast("cuda",enabled=use_amp): # pyright: ignore[reportAttributeAccessIssue]
            out=model(x)
            loss=criterion(out,y)

        if train:
            loss=loss/acc_steps
            scaler.scale(loss).backward() # pyright: ignore[reportOptionalMemberAccess]

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


def train(config_path: str):

    cfg=load_config(config_path)
    logging.basicConfig(level="INFO",format="%(asctime)s | %(message)s")
    set_seed(cfg["training"]["seed"])

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tr,pp,mp,paths=cfg["training"],cfg["preprocessing"],cfg["model"],cfg["paths"]
    os.makedirs(paths["checkpoints"],exist_ok=True)

    train_ds=BraTS_Dataset(paths["train_images"],
                           paths["train_masks"],
                           train_trf(pp["image_size"],mp["in_channels"]),
                           paths["metadata"],"train")
    
    val_ds=BraTS_Dataset(paths["val_images"],
                         paths["val_masks"],
                         val_trf(pp["image_size"],mp["in_channels"]),
                         paths["metadata"],"val")

    train_loader=DataLoader(train_ds,tr["batch_size"],True,num_workers=tr["num_workers"],pin_memory=tr["pin_memory"],drop_last=True)
    val_loader=DataLoader(val_ds,tr["batch_size"],False,num_workers=tr["num_workers"],pin_memory=tr["pin_memory"])

    model=UNet(mp["in_channels"],mp["out_channels"],mp["features"]).to(device)
    logger.info(f"Params: {count_parameters(model):,}")

    optimizer=torch.optim.Adam(model.parameters(),lr=tr["learning_rate"],weight_decay=1e-5)  # pyright: ignore[reportPrivateImportUsage]
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=5)
    scaler=torch.amp.GradScaler("cuda",enabled=tr["mixed_precision"]) # pyright: ignore[reportAttributeAccessIssue]

    mlflow.set_tracking_uri(cfg.get("mlflow",{}).get("tracking_uri","./mlruns"))
    mlflow.set_experiment(cfg.get("mlflow",{}).get("experiment_name","brats"))

    best_dice=0.0
    
    with mlflow.start_run():
        mlflow.log_params(flat(cfg))

        for epoch in range(1,tr["epochs"]+1):
            t0=time.time()

            trn=run_epoch(model,train_loader,device,tr["mixed_precision"],optimizer,scaler,tr["gradient_accumulation_steps"])
            val=run_epoch(model,val_loader,device,tr["mixed_precision"])

            scheduler.step(val["dice"])
            print(
                f"E{epoch:03d} | "
                f"T-Dice {trn['dice']:.4f} | V-Dice {val['dice']:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.5f} | "
                f"{time.time()-t0:.1f}s")\
                
            mlflow.log_metrics({
                "train_dice": trn["dice"],
                "val_dice": val["dice"],
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
                
                torch.save(checkpoint,os.path.join(paths["checkpoints"],"best.pth"))

        mlflow.log_metric("best_val_dice",best_dice)
        logger.info(f"Best Dice: {best_dice:.4f}")

