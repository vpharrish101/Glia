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
    
    for p in sorted(sub_dir.glob("*.nii*")):
        if keyword in p.name.lower():
            return p
    return None


def _normalize_slice(s: np.ndarray)->np.ndarray:

    min,max=s.min(),s.max()
    tmp=max-min
    if (tmp)<1e-8:
        return np.zeros_like(s,dtype=np.uint8)
    return ((s-min)/(tmp)*255).astype(np.uint8)


def build_context_image(vol: np.ndarray,
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


def context_slices(cfg):
    channels = cfg["model"]["in_channels"]
    context = cfg["preprocessing"].get("context_slices", 1)
    if not isinstance(context, int) or context < 0 or channels != 2 * context + 1:
        raise ValueError("model.in_channels must equal 2 * preprocessing.context_slices + 1")
    if cfg["model"]["out_channels"] != 1:
        raise ValueError("Binary segmentation requires model.out_channels=1")
    return context


def process_subject(sub_dir: Path,
                    img_size: int,
                    min_tumorpix: int,
                    slice_context: int, modality: str = "flair")->list[dict]:
    
    sub_id=sub_dir.name
    flair_pth=_find_nifti(sub_dir, modality.lower())
    seg_pth=_find_nifti(sub_dir,"seg")

    if flair_pth is None:
        logger.warning("No %s file for %s", modality, sub_id)
        return []
    if seg_pth is None:
        logger.warning(f"No segmentation file for {sub_id}")
        return []
    
    scan = ni.load(flair_pth)
    segmentation = ni.load(seg_pth)
    if len(scan.shape) != 3 or scan.shape != segmentation.shape or not np.allclose(scan.affine, segmentation.affine):
        raise ValueError(f"Image and mask must share a 3D voxel grid for {sub_id}")
    flair_data = scan.get_fdata(dtype=np.float32)
    seg_data = (segmentation.get_fdata(dtype=np.float32) > 0).astype(np.uint8)

    rec=[]
    for x in range(flair_data.shape[2]):

        tumor_pix=int(seg_data[:,:,x].sum())
        if tumor_pix<min_tumorpix:
            continue

        context_img=build_context_image(flair_data,x,slice_context,img_size)
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
    paths = cfg["paths"]
    image_dirs = {split: Path(paths[f"{split}_images"]) for split in ("train", "val")}
    mask_dirs = {split: Path(paths[f"{split}_masks"]) for split in ("train", "val")}
    img_size=cfg["preprocessing"]["image_size"]
    min_tumorpix=cfg["preprocessing"]["min_tumor_pixels"]
    val_split=cfg["preprocessing"]["val_split"]
    seed=cfg["training"]["seed"]
    context = context_slices(cfg)
    if context > 1:
        raise ValueError("NIfTI PNG preprocessing supports one or three model channels")
    modality = cfg["preprocessing"].get("modality", "flair")

    for directory in (*image_dirs.values(), *mask_dirs.values()):
        directory.mkdir(parents=True, exist_ok=True)

    subject_dirs=sorted(p for p in raw_dir.iterdir() if p.is_dir())

    if not subject_dirs:
        raise ValueError(f"No subject directories found in {raw_dir}")

    logger.info(f"Found {len(subject_dirs)} subjects in {raw_dir}")
    logger.info(f"2.5D context: {context} slices per side ({2*context+1} total channels)",)

    all_records=[]
    for subj_dir in tqdm(subject_dirs,desc="Processing subjects"):
        all_records.extend(process_subject(subj_dir, img_size, min_tumorpix, context, modality))

    if not all_records:
        raise ValueError("No valid slices extracted")

    logger.info(f"Total slices with tumor:{len(all_records)}")

    subject_ids = sorted({record["subject_id"] for record in all_records})
    if len(subject_ids) < 2:
        raise ValueError("Training needs at least two subjects with usable labeled slices")
    
    train,val=train_test_split(subject_ids,test_size=val_split,random_state=seed)
    train_subjects,val_subjects=set(train), set(val)

    logger.info(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

    metadata_rows=[]
    for rec in tqdm(all_records,desc="Saving slices"):
        split="train" if rec["subject_id"] in train_subjects else "val"
        img_path = image_dirs[split] / rec["image_filename"]
        mask_path = mask_dirs[split] / rec["mask_filename"]
        if not cv2.imwrite(str(img_path), rec["image"]) or not cv2.imwrite(str(mask_path), rec["mask"]):
            raise OSError(f"Could not write image and mask for {rec['image_filename']}")

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

    metadata_path = Path(paths["metadata"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(metadata_rows)
    df.to_csv(metadata_path,index=False)

    logger.info(f"Metadata saved to {metadata_path}")
    logger.info("Train slices: %d | Val slices: %d",
                len(df[df["split"]=="train"]),
                len(df[df["split"]=="val"])
            )
    logger.info("ETL complete.")
