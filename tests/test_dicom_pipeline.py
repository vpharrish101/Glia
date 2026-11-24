import csv
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pydicom
import yaml
from minio import Minio
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.pixels import pack_bits
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, SegmentationStorage, generate_uid

from src.ETL.series import order_slices
from src.ETL.dicom_preprocess import RejectedDicom, sanitize_DICOM
from src.ETL.segmentation import read_segmentation_masks
from src.ETL.utils.HMAC_builder import deterministic_uid

ROOT = Path(__file__).resolve().parents[1]
TEST_SECRET = "deterministic-secret-for-synthetic-test-only"


def phantom(index):
    y, x = np.indices((32, 32))
    mask = (x - 16 - index) ** 2 + (y - 15) ** 2 < 49
    pixels = (x + y + 200 * mask).astype(np.uint16)
    return pixels, mask.astype(np.uint8)


def dicom_slice(series, position, study=None, frame=None, patient_id="SYNTHETIC-123"):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    image = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    image.SOPClassUID = MRImageStorage
    image.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    image.StudyInstanceUID = study or generate_uid()
    image.SeriesInstanceUID = series
    image.FrameOfReferenceUID = frame or generate_uid()
    image.PatientID = patient_id
    image.PatientName = "Synthetic^Subject"
    image.InstitutionName = "Synthetic Hospital"
    image.SeriesDescription = "Synthetic identifying free text"
    image.add_new((0x0011, 0x1010), "LO", "Synthetic private tag")
    image.Modality = "MR"
    image.Manufacturer = "Synthetic Scanner"
    image.ImageType = ["ORIGINAL", "PRIMARY"]
    image.BurnedInAnnotation = "NO"
    image.InstanceNumber = 100 - position
    image.ImageOrientationPatient = [1, 0, 0, 0, 0, 1]
    image.ImagePositionPatient = [0, -2 * position, 0]
    image.PixelSpacing = [1, 1]
    image.set_pixel_data(phantom(position)[0], "MONOCHROME2", 12, generate_instance_uid=False)
    return image


def dicom_seg(sources, frame_sources, segment_numbers=None):
    def item(**attrs):
        result = Dataset()
        result.update(attrs)
        return result

    def reference(image):
        return item(ReferencedSOPClassUID=image.SOPClassUID, ReferencedSOPInstanceUID=image.SOPInstanceUID)

    first = sources[0]
    seg_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = seg_uid
    seg = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    seg.SOPClassUID = SegmentationStorage
    seg.SOPInstanceUID = seg_uid
    seg.Modality = "SEG"
    seg.PatientID = first.PatientID
    seg.PatientName = first.PatientName
    seg.StudyInstanceUID = first.StudyInstanceUID
    seg.SeriesInstanceUID = generate_uid()
    seg.FrameOfReferenceUID = first.FrameOfReferenceUID
    seg.SegmentationType = "BINARY"
    numbers = segment_numbers or [1] * len(frame_sources)
    seg.SegmentSequence = [item(SegmentNumber=n, SegmentLabel=f"Target {n}") for n in sorted(set(numbers))]
    seg.Rows = first.Rows
    seg.Columns = first.Columns
    seg.NumberOfFrames = len(frame_sources)
    seg.PhotometricInterpretation = "MONOCHROME2"
    seg.SamplesPerPixel = 1
    seg.BitsAllocated = seg.BitsStored = 1
    seg.HighBit = seg.PixelRepresentation = 0
    seg.ReferencedSeriesSequence = [item(
        SeriesInstanceUID=first.SeriesInstanceUID,
        ReferencedInstanceSequence=[reference(image) for image in sources],
    )]
    seg.SharedFunctionalGroupsSequence = [item(
        PlaneOrientationSequence=[item(ImageOrientationPatient=first.ImageOrientationPatient)],
        PixelMeasuresSequence=[item(PixelSpacing=first.PixelSpacing, SliceThickness=2)],
    )]
    seg.PerFrameFunctionalGroupsSequence = [
        item(
            PlanePositionSequence=[item(ImagePositionPatient=image.ImagePositionPatient)],
            SegmentIdentificationSequence=[item(ReferencedSegmentNumber=number)],
            DerivationImageSequence=[item(SourceImageSequence=[reference(image)])],
        )
        for image, number in zip(frame_sources, numbers, strict=True)
    ]
    seg.PixelData = pack_bits(np.stack([image.pixel_array > 200 for image in frame_sources]).astype(np.uint8))
    return seg


class DicomSafetyTests(unittest.TestCase):
    @patch.dict(os.environ, {"HMAC_SECRET": TEST_SECRET})
    def test_sanitization_keeps_pixels_and_geometry(self):
        image = dicom_slice(generate_uid(), 0)
        clean = sanitize_DICOM(image)
        self.assertFalse(clean.PatientName)
        self.assertNotEqual(clean.PatientID, image.PatientID)
        self.assertNotIn("SeriesDescription", clean)
        self.assertNotIn("InstitutionName", clean)
        self.assertFalse(any(element.tag.is_private for element in clean))
        self.assertNotEqual(clean.SeriesInstanceUID, image.SeriesInstanceUID)
        self.assertEqual(clean.SOPInstanceUID, clean.file_meta.MediaStorageSOPInstanceUID)
        self.assertEqual(clean.ImageOrientationPatient, image.ImageOrientationPatient)
        np.testing.assert_array_equal(clean.pixel_array, image.pixel_array)

    def test_burned_and_unknown_pixel_status_are_rejected(self):
        image = dicom_slice(generate_uid(), 0)
        for status in ("YES", ""):
            with self.subTest(status=status):
                image.BurnedInAnnotation = status
                with self.assertRaisesRegex(RejectedDicom, "BurnedInAnnotation"):
                    sanitize_DICOM(image)

    def test_duplicate_or_missing_geometry_is_rejected(self):
        series, frame = generate_uid(), generate_uid()
        one = dicom_slice(series, 0, frame=frame)
        two = dicom_slice(series, 0, frame=frame)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            order_slices([one, two])
        del two.ImagePositionPatient
        with self.assertRaisesRegex(ValueError, "ImagePositionPatient"):
            order_slices([one, two])


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        secret = patch.dict(os.environ, {"HMAC_SECRET": TEST_SECRET})
        secret.start()
        self.addCleanup(secret.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "mask.dcm"
        series, study, frame = generate_uid(), generate_uid(), generate_uid()
        self.images = [dicom_slice(series, index, study, frame) for index in range(3)]
        self.records = []
        for image in self.images:
            clean = sanitize_DICOM(image)
            self.records.append({
                "sop_uid": clean.SOPInstanceUID, "study_uid": clean.StudyInstanceUID,
                "patient_id": clean.PatientID, "frame_uid": clean.FrameOfReferenceUID,
                "series_uid": clean.SeriesInstanceUID, "rows": clean.Rows, "columns": clean.Columns,
                "image_position": clean.ImagePositionPatient, "image_orientation": clean.ImageOrientationPatient,
                "pixel_spacing_values": clean.PixelSpacing,
            })

    def masks(self, seg, **kwargs):
        seg.save_as(self.path, enforce_file_format=True)
        return read_segmentation_masks([self.path], self.records, **kwargs)

    def test_matches_reversed_frames_and_explicit_empty_slice(self):
        seg = dicom_seg(self.images, [self.images[2], self.images[0]])
        masks = self.masks(seg)
        for index in (0, 2):
            np.testing.assert_array_equal(masks[self.records[index]["sop_uid"]], phantom(index)[1])
        self.assertFalse(masks[self.records[1]["sop_uid"]].any())

        del seg.ReferencedSeriesSequence
        masks = self.masks(seg)
        self.assertNotIn(self.records[1]["sop_uid"], masks)

    def test_rejects_unknown_reference_and_shifted_geometry(self):
        seg = dicom_seg(self.images, [self.images[0]])
        frame = seg.PerFrameFunctionalGroupsSequence[0]
        reference = frame.DerivationImageSequence[0].SourceImageSequence[0]
        original_uid = reference.ReferencedSOPInstanceUID
        reference.ReferencedSOPInstanceUID = generate_uid()
        with self.assertRaisesRegex(ValueError, "absent from this batch"):
            self.masks(seg)

        reference.ReferencedSOPInstanceUID = original_uid
        frame.PlanePositionSequence[0].ImagePositionPatient = [0, 0, 2]
        with self.assertRaisesRegex(ValueError, "ImagePositionPatient does not match"):
            self.masks(seg)

    def test_requires_explicit_segment_for_multiple_labels(self):
        seg = dicom_seg(self.images, [self.images[0], self.images[1]], [1, 2])
        with self.assertRaisesRegex(ValueError, "training.segment_number"):
            self.masks(seg)
        masks = self.masks(seg, segment_number=2)
        self.assertFalse(masks[self.records[0]["sop_uid"]].any())
        np.testing.assert_array_equal(masks[self.records[1]["sop_uid"]], phantom(1)[1])


@unittest.skipUnless(os.getenv("GLIA_TEST_S3") == "1", "Set GLIA_TEST_S3=1 and configure a test S3 server")
class PipelineTests(unittest.TestCase):
    def test_train_with_png_masks_and_predict(self):
        self._train_and_predict(use_seg=False)

    def test_train_with_dicom_seg_and_predict(self):
        self._train_and_predict(use_seg=True)

    def _train_and_predict(self, use_seg):
        bucket = f"glia-test-{generate_uid().split('.')[-1][:20]}"
        env = os.environ | {
            "HMAC_SECRET": TEST_SECRET,
            "MINIO_BUCKET": bucket,
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        client = Minio(
            env["MINIO_ENDPOINT"],
            access_key=env["MINIO_ROOT_USER"],
            secret_key=env["MINIO_ROOT_PASSWORD"],
            secure=False,
        )

        def cleanup_bucket():
            if client.bucket_exists(bucket):
                for item in client.list_objects(bucket, recursive=True):
                    client.remove_object(bucket, item.object_name)
                client.remove_bucket(bucket)

        self.addCleanup(cleanup_bucket)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            env["XDG_CACHE_HOME"] = str(work / "cache")
            env["MPLCONFIGDIR"] = str(work / "cache" / "matplotlib")

            def run(*args, expected_status=0):
                result = subprocess.run(
                    [sys.executable, "-m", "src.main", *args],
                    cwd=work, env=env, text=True, capture_output=True, timeout=120,
                )
                self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
                return result.stdout + result.stderr

            cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
            cfg["preprocessing"]["image_size"] = 32
            cfg["training"].update(
                epochs=1, batch_size=16, learning_rate=0.0001,
                num_workers=2 if use_seg else 0, prefetch_factor=1,
            )
            cfg["etl"]["num_workers"] = 2
            cfg["model"]["features"] = [4, 8]
            cfg["inference"]["checkpoint"] = "checkpoints/synthetic.pth"
            cfg["paths"].update(
                dicom="incoming/dicom", masks="annotations", metadata_db="metadata/index.db",
                training_cache="training-cache",
                predictions="dicom/predictions", overlays="dicom/overlays",
            )
            config = work / "config" / "config.yaml"
            config.parent.mkdir()
            config.write_text(yaml.safe_dump(cfg))

            # Two patients are required for a patient-level validation split.
            # The first patient has two series and non-geometric filenames.
            raw = work / cfg["paths"]["dicom"]
            labels = work / cfg["paths"]["masks"]
            series, study, frame = generate_uid(), generate_uid(), generate_uid()
            images = [dicom_slice(series, index, study, frame) for index in range(3)]
            extra = dicom_slice(generate_uid(), 0, study, frame)
            other_series, other_study, other_frame = generate_uid(), generate_uid(), generate_uid()
            other = [
                dicom_slice(other_series, index, other_study, other_frame, patient_id="SYNTHETIC-456")
                for index in range(3)
            ]
            for filename, image, position in zip(
                ("first/c.dcm", "first/a.dcm", "first/b.dcm", "first/d.DICOM",
                 "second/a.dcm", "second/b.dcm", "second/c.dcm"),
                [*images, extra, *other],
                [0, 1, 2, 0, 0, 1, 2],
                strict=True,
            ):
                path = raw / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save_as(path, enforce_file_format=True)
                if not use_seg:
                    label = labels / (filename + ".png")
                    label.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(cv2.imwrite(str(label), phantom(position)[1] * 255))

            if use_seg:
                for index, series_images in enumerate([images, [extra], other]):
                    seg = dicom_seg(series_images, list(reversed(series_images)))
                    seg.save_as(raw / f"labels-{index}.dcm", enforce_file_format=True)

            training = run("train")
            self.assertIn("Staged 1 training series on disk", training)
            self.assertEqual(list((work / cfg["paths"]["training_cache"]).iterdir()), [])
            self.assertIn("V-Dice", training)
            self.assertIn("LR 0.00010", training)
            self.assertRegex(training, r"Training on 1 patients \([34] slices\), validating on 1 patients")
            checkpoint = work / cfg["inference"]["checkpoint"]
            self.assertGreater(checkpoint.stat().st_size, 0)
            with sqlite3.connect(work / cfg["paths"]["metadata_db"]) as db:
                rows = db.execute("SELECT storage_key FROM dicom_metadata").fetchall()
            self.assertEqual(len(rows), 7)

            # Inference reads the same input layout without labels.
            if not use_seg:
                labels.rename(work / "unused-labels")
            # Inference ignores SEG files even when left beside the images.
            inference_config = config.rename(work / "inference.yaml")
            self.assertNotIn("V-Dice", run("inference", "--config", str(inference_config)))
            with (work / "dicom/predictions/dicom_predictions.csv").open() as stream:
                report = list(csv.DictReader(stream))
            self.assertEqual(len(report), 7)
            with patch.dict(os.environ, {"HMAC_SECRET": TEST_SECRET}):
                clean_series = deterministic_uid(series, "series")
                expected = [deterministic_uid(image.SOPInstanceUID, "sop") for image in images]
            series_rows = [row for row in report if row["series_uid"] == clean_series]
            self.assertEqual([row["sop_uid"] for row in series_rows], expected)
            self.assertEqual([row["slice_index"] for row in series_rows], ["0", "1", "2"])
            self.assertEqual(len({row["series_uid"] for row in report}), 3)
            self.assertEqual({row[0] for row in rows}, {entry["storage_key"] for entry in report})
            self.assertEqual({item.object_name for item in client.list_objects(bucket, recursive=True)}, {row[0] for row in rows})

            for entry in report:
                local = work / "download.dcm"
                client.fget_object(bucket, entry["storage_key"], str(local))
                clean = pydicom.dcmread(local)
                self.assertFalse(clean.PatientName)
                self.assertNotIn("SeriesDescription", clean)
                self.assertNotIn(clean.PatientID, {"SYNTHETIC-123", "SYNTHETIC-456"})
                self.assertEqual(clean.SOPInstanceUID, entry["sop_uid"])
                mask = cv2.imread(str(work / entry["prediction"]), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(mask.shape, (32, 32))
                self.assertTrue(np.isin(mask, [0, 255]).all())
                self.assertEqual(cv2.imread(str(work / entry["overlay"])).shape, (32, 32, 3))
                local.unlink()

            images[0].save_as(raw / "first/duplicate.dcm", enforce_file_format=True)
            cfg["paths"]["predictions"] = "rejected/predictions"
            inference_config.write_text(yaml.safe_dump(cfg))
            log = run("inference", "--config", str(inference_config), expected_status=1)
            self.assertIn("Duplicate SOPInstanceUID", log)
            self.assertFalse((work / cfg["paths"]["predictions"]).exists())
