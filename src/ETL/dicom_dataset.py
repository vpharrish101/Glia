"""Read sanitized DICOM objects as ordered 2.5D segmentation inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut
from torch.utils.data import Dataset

from src.ETL.dataclass import val_trf
from src.ETL.preprocessing import build_context_image


def _geometry_vector(image, field, size):
    values = np.asarray(image.get(field, []), dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"Missing or invalid {field} in series {image.SeriesInstanceUID}")
    return values


def order_slices(images):
    if len(images) == 1:
        return images

    first = images[0]
    series_uid = first.SeriesInstanceUID
    row, column = _geometry_vector(first, "ImageOrientationPatient", 6).reshape(2, 3)
    if not np.allclose([np.linalg.norm(row), np.linalg.norm(column), row @ column], [1, 1, 0], atol=1e-4):
        raise ValueError(f"Invalid orientation in series {series_uid}")
    normal = np.cross(row, column)
    origin = _geometry_vector(first, "ImagePositionPatient", 3)
    spacing = _geometry_vector(first, "PixelSpacing", 2)
    if (spacing <= 0).any():
        raise ValueError(f"Invalid pixel spacing in series {series_uid}")

    positions = []
    for image in images:
        orientation = _geometry_vector(image, "ImageOrientationPatient", 6)
        pixel_spacing = _geometry_vector(image, "PixelSpacing", 2)
        if (
            not np.allclose(orientation, np.concatenate([row, column]), atol=1e-4)
            or not np.allclose(pixel_spacing, spacing, atol=1e-4)
            or (image.Rows, image.Columns) != (first.Rows, first.Columns)
            or image.get("FrameOfReferenceUID") != first.get("FrameOfReferenceUID")
            or image.get("EchoNumbers") != first.get("EchoNumbers")
            or image.get("TemporalPositionIdentifier") != first.get("TemporalPositionIdentifier")
        ):
            raise ValueError(f"Inconsistent image geometry or acquisition in series {series_uid}")
        offset = _geometry_vector(image, "ImagePositionPatient", 3) - origin
        distance = float(offset @ normal)
        if not np.allclose(offset, distance * normal, atol=0.1):
            raise ValueError(f"Slice grids are offset within series {series_uid}")
        positions.append((distance, image))

    positions.sort(key=lambda item: item[0])
    gaps = np.diff([distance for distance, _ in positions])
    if (gaps <= 1e-3).any() or not np.allclose(gaps, gaps[0], atol=0.1, rtol=0.01):
        raise ValueError(f"Duplicate or irregular slice positions in series {series_uid}")
    return [image for _, image in positions]


class DicomDataset(Dataset):
    def __init__(self, lake, records, image_size, context_slices):
        images = []
        with TemporaryDirectory() as directory:
            path = Path(directory) / "slice.dcm"
            for record in records:
                lake.get(record["storage_key"], path)
                image = pydicom.dcmread(path)
                tags = (image.PatientID, image.StudyInstanceUID, image.SeriesInstanceUID, image.SOPInstanceUID)
                expected = tuple(record[key] for key in ("patient_id", "study_uid", "series_uid", "sop_uid"))
                if image.PatientIdentityRemoved != "YES" or tags != expected:
                    raise ValueError("Stored DICOM does not match the sanitized ingestion record")
                images.append(image)
                path.unlink()

        ordered = order_slices(images)
        slices = []
        for image in ordered:
            pixels = apply_modality_lut(image.pixel_array, image).astype(np.float32)
            if not np.isfinite(pixels).all():
                raise ValueError(f"Invalid pixel values in slice {image.SOPInstanceUID}")
            if image.PhotometricInterpretation == "MONOCHROME1":
                pixels = -pixels
            slices.append(pixels)

        self.volume = np.stack(slices, axis=-1)
        self.sop_uids = [image.SOPInstanceUID for image in ordered]
        self.image_size = image_size
        self.context_slices = context_slices
        self.transform = val_trf(image_size, 2 * context_slices + 1)

    def __len__(self):
        return len(self.sop_uids)

    def __getitem__(self, index):
        image = build_context_image(self.volume, index, self.context_slices, self.image_size)
        return {
            "image": self.transform(image=image)["image"].float(),
            "filename": f"{self.sop_uids[index]}.png",
        }
