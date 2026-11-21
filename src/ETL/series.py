"""Read sanitized DICOM objects in physical slice order."""

from collections import defaultdict

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut


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


def read_series(lake, records):
    """Yield each pixel volume and its sanitized records in matching slice order."""
    groups = defaultdict(list)
    for record in records:
        key = (record["patient_id"], record["study_uid"], record["series_uid"])
        groups[key].append(record)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "slice.dcm"
        for _, entries in sorted(groups.items()):
            yield _read_volume(lake, path, entries)


def _read_volume(lake, path, records):
    images = []
    by_uid = {}
    for record in records:
        lake.get(record["storage_key"], path)
        image = pydicom.dcmread(path)
        path.unlink()
        tags = (image.PatientID, image.StudyInstanceUID, image.SeriesInstanceUID, image.SOPInstanceUID)
        expected = tuple(record[key] for key in ("patient_id", "study_uid", "series_uid", "sop_uid"))
        if image.PatientIdentityRemoved != "YES" or tags != expected:
            raise ValueError("Stored DICOM does not match the sanitized ingestion record")
        images.append(image)
        by_uid[image.SOPInstanceUID] = record

    slices = []
    ordered = order_slices(images)
    for image in ordered:
        pixels = apply_modality_lut(image.pixel_array, image).astype(np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError(f"Invalid pixel values in slice {image.SOPInstanceUID}")
        if image.PhotometricInterpretation == "MONOCHROME1":
            pixels = -pixels
        slices.append(pixels)

    return np.stack(slices, axis=-1), [by_uid[image.SOPInstanceUID] for image in ordered]
