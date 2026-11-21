"""Prepare single-frame monochrome DICOM images for the data lake."""

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import pydicom
from deid.config import DeidRecipe
from deid.dicom.pixels.detect import has_burned_pixels
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, UID

from src.ETL.Object_storage import DataLake
from src.ETL.utils.HMAC_builder import deterministic_uid, patient_hmac
from src.ETL.utils.logging_mod import get_logger

logger = get_logger("DICOM")
_worker_lake = None
SUPPORTED_STORAGE = {MRImageStorage: "MR", CTImageStorage: "CT"}
PIXEL_RECIPE = Path(__file__).resolve().parents[2] / "config" / "deid.dicom"
# Keep the numeric image geometry and acquisition fields used by readers.
# Text, private elements, sequences and overlays are omitted by construction.
IMAGE_FIELDS = (
    "SeriesNumber", "InstanceNumber", "ImagePositionPatient",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
    "SpacingBetweenSlices", "EchoTime", "EchoNumbers", "RepetitionTime",
    "InversionTime", "MagneticFieldStrength", "FlipAngle", "EchoTrainLength",
    "NumberOfAverages", "TemporalPositionIdentifier", "RescaleSlope",
    "RescaleIntercept", "KVP",
)
EMPTY_FIELDS = (
    "PatientName", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
    "ReferringPhysicianName", "StudyID", "AccessionNumber", "PositionReferenceIndicator",
)


class RejectedDicom(ValueError):
    """Input outside the supported DICOM or pixel policy."""


@lru_cache(maxsize=1)
def pixel_recipe():
    return DeidRecipe(str(PIXEL_RECIPE))


def sanitize_DICOM(source):
    storage_class = source.get("SOPClassUID")
    modality = SUPPORTED_STORAGE.get(storage_class)
    if modality is None or source.get("Modality") != modality:
        raise RejectedDicom("Expected classic MR or CT Image Storage")
    photometric = source.get("PhotometricInterpretation")
    if (
        int(source.get("NumberOfFrames", 1)) != 1
        or source.get("SamplesPerPixel") != 1
        or photometric not in {"MONOCHROME1", "MONOCHROME2"}
        or "PixelData" not in source
    ):
        raise RejectedDicom("Expected single-frame monochrome pixel data")
    if (
        str(source.get("BurnedInAnnotation", "")).strip().upper() != "NO"
        or str(source.get("RecognizableVisualFeatures", "")).strip().upper() == "YES"
    ):
        raise RejectedDicom("Input must declare BurnedInAnnotation=NO and have no flagged recognizable features")
    image_type = source.get("ImageType", [])
    if isinstance(image_type, str) or len(image_type) < 2:
        raise RejectedDicom("Missing or invalid ImageType")

    patient_id = str(source.get("PatientID", "")).strip()
    if not patient_id:
        raise RejectedDicom("Missing PatientID")
    uid_fields = {
        "StudyInstanceUID": "study",
        "SeriesInstanceUID": "series",
        "SOPInstanceUID": "sop",
        "FrameOfReferenceUID": "frame",
    }
    mapped = {}
    for field, namespace in uid_fields.items():
        value = UID(str(source.get(field, "")))
        if value.is_valid:
            mapped[field] = deterministic_uid(value, namespace)
        elif field != "FrameOfReferenceUID" or value:
            raise RejectedDicom(f"Missing or invalid {field}")

    result = has_burned_pixels(source, force=False, deid=pixel_recipe())
    if any(hit["group"] in {"blacklist", "graylist"} for hit in result["results"]):
        raise RejectedDicom("Pixel rules flagged the image for review")

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = storage_class
    file_meta.MediaStorageSOPInstanceUID = mapped["SOPInstanceUID"]
    clean = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    clean.SOPClassUID = storage_class
    clean.Modality = modality
    for field, value in mapped.items():
        setattr(clean, field, value)
    for field in IMAGE_FIELDS:
        if field in source:
            clean.add(source[field])
    for field in EMPTY_FIELDS:
        setattr(clean, field, "")
    clean.ImageType = [
        "ORIGINAL" if image_type[0] == "ORIGINAL" else "DERIVED",
        "PRIMARY" if image_type[1] == "PRIMARY" else "SECONDARY",
    ]
    clean.PatientID = patient_hmac(patient_id)
    clean.PatientIdentityRemoved = "YES"
    clean.DeidentificationMethod = "Glia metadata allowlist and UID pseudonymization"
    clean.BurnedInAnnotation = "NO"
    clean.set_pixel_data(source.pixel_array, photometric, source.BitsStored, generate_instance_uid=False)
    return clean


def processDICOM(source_file, lake=None):
    try:
        clean = sanitize_DICOM(pydicom.dcmread(source_file))
        sop_uid = clean.SOPInstanceUID
        if lake is None:
            lake = DataLake()
        object_name = f"{clean.PatientID}_{sop_uid}.dcm"
        with TemporaryDirectory() as directory:
            path = Path(directory) / object_name
            clean.save_as(path, enforce_file_format=True)
            storage_key = lake.put(path, f"dicom/{object_name}")
        return {
            "filename": object_name,
            "patient_id": clean.PatientID,
            "status": "pixel_policy_passed",
            "storage_key": storage_key,
            "modality": clean.Modality,
            "study_uid": clean.StudyInstanceUID,
            "series_uid": clean.SeriesInstanceUID,
            "sop_uid": clean.SOPInstanceUID,
            "manufacturer": None,
            "manufacturer_model": None,
            "body_part": None,
            "rows": clean.Rows,
            "columns": clean.Columns,
            "pixel_spacing": str(clean.get("PixelSpacing", "")),
            "slice_thickness": str(clean.get("SliceThickness", "")),
            "patient_sex": None,
            "patient_age": None,
            "processing_status": "success",
        }
    except RejectedDicom as error:
        return {"status": "error", "reason": str(error)}
    except Exception as error:
        # Library exceptions may contain identifying filenames or tag values.
        return {"status": "error", "reason": f"DICOM read, decode or storage failed ({type(error).__name__})"}


def _initialize_worker():
    global _worker_lake
    # The parent has already prepared the bucket. Each process owns its client.
    _worker_lake = DataLake(ensure_bucket=False)


def _process_in_worker(path):
    return processDICOM(path, _worker_lake)


def processDICOMBatch(source_dir, lake, num_workers=4):
    source_dir = Path(source_dir)
    paths = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )
    logger.info("DICOM files discovered: %d", len(paths))
    # Reject duplicate object identities before workers can upload the same key.
    seen_sops = set()
    for path in paths:
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        except (OSError, ValueError, EOFError, pydicom.errors.InvalidDicomError):
            continue  # The processing pass reports unreadable files.
        uid = str(header.get("SOPInstanceUID", ""))
        if uid and uid in seen_sops:
            raise RejectedDicom("Duplicate SOPInstanceUID in this batch")
        seen_sops.add(uid)

    if num_workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(paths)),
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
        ) as executor:
            records = list(executor.map(_process_in_worker, paths))
    else:
        records = [processDICOM(path, lake) for path in paths]

    for index, (path, record) in enumerate(zip(paths, records), start=1):
        if record["status"] == "error":
            logger.warning("DICOM input %d rejected: %s", index, record["reason"])
        else:
            # Keep the source path in memory only so the caller can match input labels.
            record["source_path"] = path.relative_to(source_dir)
    return records
