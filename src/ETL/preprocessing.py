import sys
import logging
import cv2
import yaml
import nibabel as ni
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

logger=logging.getLogger(__name__)

def _find_nifti(sub_dir: Path,
               keyword: str)->Path|None:
    
    for p in sub_dir.glob("*.nii*"):
        if keyword in p.name.lower():
            return p
    return None


def _normalize_slice(s: np.ndarray)->np.ndarray:

    min,max=s.min(),s.max()
    tmp=max-min
    if (tmp)<1e-8:
        return np.zeros_like(s,dtype=np.uint8)
    return ((s-min)/(tmp)*255).astype(np.uint8)


def _build_ctximg(vol: np.ndarray,
                  center: int,
                  context: int,
                  img_size: int)-> np.ndarray:
    
    slices=vol.shape[2]
    channels=[]
    for offset in range(-context,context+1):
        idx=center+offset
        if 0<=idx<slices:
            ch=_normalize_slice(vol[:,:,idx])
        else: 
            ch=np.zeros((vol.shape[0],vol.shape[1]),dtype=np.uint8)
        ch=cv2.resize(ch,(img_size,img_size),interpolation=cv2.INTER_LINEAR)
        channels.append(ch)
    return np.stack(channels,axis=-1)


def process_subject(sub_dir: Path,
                    img_size: int,
                    min_tumorpix: int,
                    slice_context: int)->list[dict]:
    
    sub_id=sub_dir.name
    flair_pth=_find_nifti(sub_dir, "flair")
    seg_pth=_find_nifti(sub_dir,"seg")

    if flair_pth is None:
        logger.warning(f"No FLAIR file for {sub_id}")
        return []
    if seg_pth is None:
        logger.warning(f"No segmentation file for {sub_id}")
        return []
    
    flair_data=ni.load(str(flair_pth)).get_fdata(dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    seg_data=(ni.load(str(seg_pth)).get_fdata(dtype=np.float32)>0).astype(np.uint8) # pyright: ignore[reportAttributeAccessIssue]

    rec=[]
    for x in range(flair_data.shape[2]):

        tumor_pix=int(seg_data[:,:,x].sum())
        if tumor_pix<min_tumorpix:
            continue

        context_img=_build_ctximg(flair_data,x,slice_context,img_size)
        mask=cv2.resize(seg_data[:,:,x],(img_size,img_size),interpolation=cv2.INTER_NEAREST)

        mask=(mask*255).astype(np.uint8)
        fname=f"{sub_id}_slice{x:04d}"

        rec.append({
            "subject_id": sub_id,
            "slice_idx": x,
            "image_filename": f"{fname}.png",
            "mask_filename": f"{fname}_mask.png",
            "tumor_pixels": tumor_pix,
            "image": context_img,
            "mask": mask,
        })

    return rec


def run_etl(config_path: str):
    cfg=yaml.safe_load(Path(config_path).read_text())

    logging.basicConfig(level=cfg.get("logging",{}).get("level","INFO"))

    raw_dir=Path(cfg["paths"]["raw_data"])
    processed_dir=Path(cfg["paths"]["processed_data"])
    img_size=cfg["preprocessing"]["image_size"]
    min_tumorpix=cfg["preprocessing"]["min_tumor_pixels"]
    val_split=cfg["preprocessing"]["val_split"]
    seed=cfg["training"]["seed"]
    context_slices=cfg["preprocessing"].get("context_slices",1)

    for split in ("train","val"):
        (processed_dir/"images"/split).mkdir(parents=True,exist_ok=True)
        (processed_dir/"masks"/split).mkdir(parents=True,exist_ok=True)

    subject_dirs=sorted(p for p in raw_dir.iterdir() if p.is_dir())

    if not subject_dirs:
        logger.error(f"No subject directories found in {raw_dir}")
        sys.exit(1)

    logger.info(f"Found {len(subject_dirs)} subjects in {raw_dir}")
    logger.info(f"2.5D context: {context_slices} slices per side ({2*context_slices+1} total channels)",)

    all_records=[]
    for subj_dir in tqdm(subject_dirs,desc="Processing subjects"):
        all_records.extend(process_subject(subj_dir,img_size,min_tumorpix,context_slices))

    if not all_records:
        logger.error("No valid slices extracted.")
        sys.exit(1)

    logger.info(f"Total slices with tumor:{len(all_records)}")

    subject_ids=list({r["subject_id"] for r in all_records})
    
    train,val=train_test_split(subject_ids,test_size=val_split,random_state=seed)
    train_subjects,val_subjects=set(train), set(val)

    logger.info(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

    metadata_rows=[]
    for rec in tqdm(all_records,desc="Saving slices"):
        split="train" if rec["subject_id"] in train_subjects else "val"
        img_path=processed_dir/"images"/split/rec["image_filename"]
        mask_path=processed_dir/"masks"/split/rec["mask_filename"]

        cv2.imwrite(str(img_path),rec["image"])
        cv2.imwrite(str(mask_path),rec["mask"])

        metadata_rows.append({
            "subject_id": rec["subject_id"],
            "slice_idx": rec["slice_idx"],
            "split": split,
            "image_filename": rec["image_filename"],
            "mask_filename": rec["mask_filename"],
            "tumor_pixels": rec["tumor_pixels"],
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            })

    metadata_path=processed_dir/"metadata.csv"
    df=pd.DataFrame(metadata_rows)
    df.to_csv(metadata_path,index=False)

    logger.info(f"Metadata saved to {metadata_path}")
    logger.info("Train slices: %d | Val slices: %d",
                len(df[df["split"]=="train"]),
                len(df[df["split"]=="val"])
            )
    logger.info("ETL complete.")
