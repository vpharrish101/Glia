"""Match binary DICOM SEG masks to sanitized single-frame image records."""

import numpy as np
import pydicom
from pydicom.uid import SegmentationStorage, UID

from src.ETL.utils.HMAC_builder import deterministic_uid, patient_hmac


def _mapped_uid(value, namespace):
    uid = UID(str(value))
    if not uid.is_valid:
        raise ValueError("DICOM SEG has a missing or invalid reference UID")
    return deterministic_uid(uid, namespace)


def _source_record(uid, sources, seg, study_uid, patient_id, frame_uid):
    source = sources.get(uid)
    if source is None:
        raise ValueError(f"DICOM SEG references a slice absent from this batch: {uid}")
    if source["study_uid"] != study_uid or (patient_id and source["patient_id"] != patient_id):
        raise ValueError(f"DICOM SEG patient or study does not match slice {uid}")
    if frame_uid and source["frame_uid"] != frame_uid:
        raise ValueError(f"DICOM SEG frame of reference does not match slice {uid}")
    if (seg.Rows, seg.Columns) != (source["rows"], source["columns"]):
        raise ValueError(f"DICOM SEG dimensions do not match slice {uid}")
    return source


def _validate_grid(source, frame, shared, reference):
    preserved = reference.get("SpatialLocationsPreserved")
    if preserved not in (None, "", "YES"):
        raise ValueError("DICOM SEG must preserve source pixel locations")

    geometry = (
        ("PlanePositionSequence", "ImagePositionPatient", "image_position", 3, 0.01),
        ("PlaneOrientationSequence", "ImageOrientationPatient", "image_orientation", 6, 0.0001),
        ("PixelMeasuresSequence", "PixelSpacing", "pixel_spacing_values", 2, 0.0001),
    )
    for sequence, tag, key, size, tolerance in geometry:
        group = frame.get(sequence) or shared.get(sequence)
        value = group[0].get(tag) if group else None
        if value is None and preserved == "YES":
            continue
        actual = np.asarray(value, dtype=float)
        expected = np.asarray(source[key], dtype=float)
        if actual.shape != (size,) or expected.shape != (size,) or not np.allclose(
            actual, expected, atol=tolerance, rtol=0,
        ):
            raise ValueError(f"DICOM SEG {tag} does not match slice {source['sop_uid']}")


def _read_masks(seg, sources, segment_number):
    if (
        seg.get("SOPClassUID") != SegmentationStorage
        or seg.get("SegmentationType") != "BINARY"
        or seg.get("BitsAllocated") != 1
        or seg.get("SamplesPerPixel") != 1
    ):
        raise ValueError("Expected a BINARY DICOM SEG object")
    segments = {int(item.SegmentNumber) for item in seg.get("SegmentSequence", [])}
    if segment_number is None:
        if len(segments) != 1:
            raise ValueError("DICOM SEG needs one segment or an explicit training.segment_number")
        segment_number = next(iter(segments))
    elif segment_number not in segments:
        raise ValueError(f"DICOM SEG does not contain segment {segment_number}")

    study_uid = _mapped_uid(seg.get("StudyInstanceUID", ""), "study")
    patient_id = str(seg.get("PatientID", "")).strip()
    patient_id = patient_hmac(patient_id) if patient_id else None
    frame_uid = _mapped_uid(seg.FrameOfReferenceUID, "frame") if seg.get("FrameOfReferenceUID") else None
    shared = (seg.get("SharedFunctionalGroupsSequence") or [{}])[0]
    frames = seg.get("PerFrameFunctionalGroupsSequence", [])
    pixels = seg.pixel_array.reshape(-1, seg.Rows, seg.Columns)
    if len(frames) != len(pixels) or not len(frames):
        raise ValueError("DICOM SEG needs functional groups for every frame")

    masks = {}
    for frame, mask in zip(frames, pixels, strict=True):
        segment = frame.get("SegmentIdentificationSequence") or shared.get("SegmentIdentificationSequence")
        if not segment or len(segment) != 1:
            raise ValueError("DICOM SEG frame needs a segment identifier")
        if int(segment[0].ReferencedSegmentNumber) != segment_number:
            continue
        references = [
            image for derivation in (frame.get("DerivationImageSequence") or shared.get("DerivationImageSequence", []))
            for image in derivation.get("SourceImageSequence", [])
        ]
        if len(references) != 1 or references[0].get("ReferencedFrameNumber", 1) != 1:
            raise ValueError("DICOM SEG needs exactly one single-frame source image per mask frame")
        reference = references[0]
        uid = _mapped_uid(reference.get("ReferencedSOPInstanceUID", ""), "sop")
        source = _source_record(uid, sources, seg, study_uid, patient_id, frame_uid)
        if uid in masks:
            raise ValueError(f"Multiple DICOM SEG frames label slice {uid}")
        _validate_grid(source, frame, shared, reference)
        masks[uid] = mask

    # Many SEG writers omit empty frames but explicitly list their source instances.
    # A slice without a frame or an explicit source reference remains unlabeled.
    for series in seg.get("ReferencedSeriesSequence", []):
        series_uid = _mapped_uid(series.get("SeriesInstanceUID", ""), "series")
        for reference in series.get("ReferencedInstanceSequence", []):
            uid = _mapped_uid(reference.get("ReferencedSOPInstanceUID", ""), "sop")
            source = _source_record(uid, sources, seg, study_uid, patient_id, frame_uid)
            if series_uid != source["series_uid"]:
                raise ValueError(f"DICOM SEG series reference does not match slice {uid}")
            if uid not in masks:
                masks[uid] = np.zeros((source["rows"], source["columns"]), dtype=np.uint8)

    if len({sources[uid]["series_uid"] for uid in masks}) != 1:
        raise ValueError("DICOM SEG must label slices from exactly one source series")
    return masks


def read_segmentation_masks(paths, records, segment_number=None):
    """Return slice masks keyed by the same pseudonymous UID as the scan ETL."""
    if segment_number is not None and (type(segment_number) is not int or segment_number < 1):
        raise ValueError("training.segment_number must be a positive integer")

    sources = {record["sop_uid"]: record for record in records}
    masks = {}
    for index, path in enumerate(paths, start=1):
        try:
            labels = _read_masks(pydicom.dcmread(path), sources, segment_number)
        except ValueError as error:
            raise ValueError(f"DICOM SEG {index}: {error}") from error
        except (AttributeError, OSError, RuntimeError, pydicom.errors.InvalidDicomError) as error:
            raise ValueError(f"DICOM SEG {index} could not be read or decoded") from error
        if masks.keys() & labels.keys():
            raise ValueError(f"DICOM SEG {index} overlaps annotations from another SEG file")
        masks.update(labels)
    return masks
