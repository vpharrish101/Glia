import cv2
import torch
import logging
import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

logger=logging.getLogger(__name__)

def train_trf(img_size: int=256,
                      num_channels: int=3) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.474),
            A.VerticalFlip(p=0.645),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussianBlur(blur_limit=(3,5),p=0.2),
            A.Resize(img_size,img_size),
            A.Normalize(
                mean=(0.0,)*num_channels,
                std=(1.0,)*num_channels
            ),
            ToTensorV2(),
        ]
    )


def val_trf(image_size: int=256,
                       num_channels: int=3)->A.Compose:
    return A.Compose([
        A.Resize(image_size,image_size),
        A.Normalize(
                mean=(0.0,)*num_channels,
                std=(1.0,)*num_channels
            ),
        ToTensorV2(),
        ]
    )

class BraTS_Dataset(Dataset):

    def __init__(self,
                 img_dir:str,
                 mask_dir:str,
                 transform:Optional[Callable]=None,
                 meta_csv: Optional[str]=None,
                 split: Optional[str]=None,):
        
        self.img_dir=Path(img_dir)
        self.mask_dir=Path(mask_dir)
        self.transform=transform

        if meta_csv and Path(meta_csv).exists() and split:
            df=pd.read_csv(meta_csv)
            df_split=df[df["split"]==split]
            self.pairs = list(
                df_split.sort_values("image_filename")[["image_filename", "mask_filename"]].itertuples(index=False, name=None)
            )

        else:
            images = sorted(self.img_dir.glob("*.png"))
            self.pairs = [(path.name, f"{path.stem}_mask.png") for path in images]
            # Older processed datasets can use the image basename for both files.
            self.pairs = [
                (image, mask if (self.mask_dir / mask).is_file() else image)
                for image, mask in self.pairs
            ]
        logger.info("Dataset [%s]: %d samples", split or "all", len(self.pairs))
    
    def __len__(self)->int:
        return len(self.pairs)

    def __getitem__(self, 
                    idx: int)->dict:
        
        image_name, mask_name = self.pairs[idx]
        img_path = self.img_dir / image_name
        mask_path = self.mask_dir / mask_name

        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if image.ndim == 2:
            image = image[..., None]
        mask=(mask>127).astype(np.float32)

        if self.transform:
            augmented=self.transform(image=image,mask=mask)
            image=augmented["image"]   
            mask=augmented["mask"]    

        else:
            image=image.astype(np.float32)/255.0
            image=torch.from_numpy(image).permute(2,0,1) 
            mask=torch.from_numpy(mask)

        if mask.ndim==2:
            mask=mask.unsqueeze(0)

        return {
            "image": image.float(),
            "mask": mask.float(),
            "filename": image_name,
            }
