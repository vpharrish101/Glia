"""Prepare ordered image stacks for binary segmentation."""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scan:
    patient_id: str
    slice_ids: list[str]
    images: np.ndarray  # height, width, slices
    masks: np.ndarray | None = None


@dataclass(frozen=True)
class StagedScan:
    patient_id: str
    slice_ids: list[str]
    images_path: Path
    masks_path: Path


def stage_training_scans(scans, directory):
    for count, scan in enumerate(scans, start=1):
        if scan.masks is None:
            raise ValueError("Training requires a mask for every DICOM slice")
        images_path = directory / f"images-{count}.npy"
        masks_path = directory / f"masks-{count}.npy"
        # Store complete slices contiguously for efficient random batch reads.
        np.save(images_path, np.ascontiguousarray(np.moveaxis(scan.images, -1, 0)))
        np.save(masks_path, np.ascontiguousarray(np.moveaxis(scan.masks, -1, 0)))
        if count == 1 or count % 25 == 0:
            logger.info("Staged %d training series on disk", count)
        yield StagedScan(scan.patient_id, scan.slice_ids, images_path, masks_path)


def prepare_scan(patient_id, slice_ids, images, image_size, mask_paths=None, *, mask_pixels=None):
    target_size = (image_size, image_size)
    masks = None
    labels = mask_paths if mask_paths is not None else mask_pixels
    if labels is not None:
        mask_slices = []
        for slice_id, mask in zip(slice_ids, labels, strict=True):
            if mask_paths is not None:
                if not mask.is_file():
                    raise ValueError(f"Missing PNG mask for slice {slice_id}")
                mask = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != images.shape[:2]:
                raise ValueError(f"Mask for slice {slice_id} must be grayscale and match the DICOM pixel grid")
            mask_slices.append(cv2.resize((mask > 0).astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST))
        masks = np.stack(mask_slices, axis=-1)

    image_slices = []
    for index in range(images.shape[2]):
        image = images[:, :, index]
        low, high = image.min(), image.max()
        if high - low < 1e-8:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            image = ((image - low) / (high - low) * 255).astype(np.uint8)
        image_slices.append(cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR))
    return Scan(patient_id, slice_ids, np.stack(image_slices, axis=-1), masks)


class SliceDataset(Dataset):
    def __init__(self, scans, cfg, augment=False):
        self.context = cfg["preprocessing"].get("context_slices", 1)
        channels = cfg["model"]["in_channels"]
        if not isinstance(self.context, int) or self.context < 0 or channels != 2 * self.context + 1:
            raise ValueError("model.in_channels must equal 2 * preprocessing.context_slices + 1")
        if cfg["model"]["out_channels"] != 1:
            raise ValueError("Binary segmentation requires model.out_channels=1")

        transforms = []
        if augment:
            aug = cfg.get("augmentation", {})
            transforms.extend([
                A.HorizontalFlip(p=aug.get("horizontal_flip_p", 0.474)),
                A.VerticalFlip(p=aug.get("vertical_flip_p", 0.645)),
                A.RandomRotate90(p=aug.get("rotate_90_p", 0.5)),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, **aug.get(
                    "shift_scale_rotate", {"shift_limit": 0.05, "scale_limit": 0.1, "rotate_limit": 15, "p": 0.5},
                )),
                A.RandomBrightnessContrast(**aug.get(
                    "brightness_contrast", {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.3},
                )),
                A.GaussianBlur(**aug.get("gaussian_blur", {"blur_limit": (3, 5), "p": 0.2})),
            ])
        self.transform = A.Compose([
            *transforms,
            A.Normalize(mean=(0.0,) * channels, std=(1.0,) * channels),
            ToTensorV2(),
        ])
        self.samples = [(scan, index) for scan in scans for index in range(len(scan.slice_ids))]
        self._mapped_arrays = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def _arrays(self, scan):
        if isinstance(scan, Scan):
            return scan.images, scan.masks
        key = scan.images_path
        arrays = self._mapped_arrays.pop(key, None)
        if arrays is None:
            arrays = (
                np.load(key, mmap_mode="r").transpose(1, 2, 0),
                np.load(scan.masks_path, mmap_mode="r").transpose(1, 2, 0),
            )
        self._mapped_arrays[key] = arrays
        if len(self._mapped_arrays) > 4:
            self._mapped_arrays.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        scan, center = self.samples[index]
        images, masks = self._arrays(scan)
        blank = np.zeros(images.shape[:2], dtype=np.uint8)
        channels = [
            images[:, :, neighbor] if 0 <= neighbor < len(scan.slice_ids) else blank
            for neighbor in range(center - self.context, center + self.context + 1)
        ]
        image = np.stack(channels, axis=-1)
        inputs = {"image": image}
        if masks is not None:
            inputs["mask"] = masks[:, :, center].copy()
        sample = self.transform(**inputs)
        sample["image"] = sample["image"].float()
        if "mask" in sample:
            sample["mask"] = sample["mask"].unsqueeze(0).float()
        sample["slice_id"] = scan.slice_ids[center]
        return sample
