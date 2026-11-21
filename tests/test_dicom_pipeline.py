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
import nibabel as nib
import numpy as np
import pydicom
import pymupdf
import yaml
from minio import Minio
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.ETL.dicom_dataset import order_slices
from src.ETL.dicom_preprocess import RejectedDicom, sanitize_DICOM
from src.ETL.utils.HMAC_builder import deterministic_uid

ROOT = Path(__file__).resolve().parents[1]
TEST_SECRET = "deterministic-secret-for-synthetic-test-only"


def phantom(index):
    y, x = np.indices((32, 32))
    mask = (x - 16 - index) ** 2 + (y - 15) ** 2 < 49
    pixels = (x + y + 200 * mask).astype(np.uint16)
    return pixels, mask.astype(np.uint8)


def dicom_slice(series, position, study=None, frame=None):
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
    image.PatientID = "SYNTHETIC-123"
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


@unittest.skipUnless(os.getenv("GLIA_TEST_S3") == "1", "Set GLIA_TEST_S3=1 and configure a test S3 server")
class PipelineTests(unittest.TestCase):
    def test_trained_checkpoint_predicts_sanitized_dicom_series(self):
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

            def run(*args, module="src.main", expected_status=0):
                result = subprocess.run(
                    [sys.executable, "-m", module, *args],
                    cwd=work, env=env, text=True, capture_output=True, timeout=120,
                )
                self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)
                return result.stdout

            cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
            cfg["preprocessing"]["image_size"] = 32
            cfg["training"].update(epochs=1, batch_size=16, num_workers=0)
            cfg["model"]["features"] = [4, 8]
            cfg["inference"]["checkpoint"] = "checkpoints/synthetic.pth"
            cfg["paths"]["metadata"] = "custom/training.csv"
            (work / "config.yaml").write_text(yaml.safe_dump(cfg))
            pixels, masks = (np.stack(parts, axis=-1) for parts in zip(*(phantom(index) for index in range(3))))
            for index in range(2):
                subject = work / "data" / "raw" / f"synthetic-{index}"
                subject.mkdir(parents=True)
                nib.save(nib.Nifti1Image(pixels, np.eye(4)), subject / "flair.nii.gz")
                nib.save(nib.Nifti1Image(masks, np.eye(4)), subject / "seg.nii.gz")
            self.assertIn("Samples: 3", run("train"))
            self.assertTrue((work / cfg["inference"]["checkpoint"]).is_file())

            # File names and InstanceNumber disagree with physical slice order.
            raw = work / "data" / "DICOMs"
            raw.mkdir(parents=True)
            series, study, frame = generate_uid(), generate_uid(), generate_uid()
            images = [dicom_slice(series, index, study, frame) for index in range(3)]
            extra = dicom_slice(generate_uid(), 0, study, frame)
            for label, image in zip(("c.dcm", "a.dcm", "b.dcm", "d.DICOM"), [*images, extra]):
                image.save_as(raw / label, enforce_file_format=True)
            etl_cfg = work / "clinical.yaml"
            etl_cfg.write_text((ROOT / "config" / "config.yaml").read_text())

            cfg["paths"].update(predictions="dicom/predictions", overlays="dicom/overlays")
            (work / "config.yaml").write_text(yaml.safe_dump(cfg))
            self.assertNotIn("Mean Dice", run("dicom", "--etl-config", str(etl_cfg)))
            with (work / "dicom/predictions/dicom_predictions.csv").open() as stream:
                report = list(csv.DictReader(stream))
            self.assertEqual(len(report), 4)
            with patch.dict(os.environ, {"HMAC_SECRET": TEST_SECRET}):
                clean_series = deterministic_uid(series, "series")
                expected = [deterministic_uid(image.SOPInstanceUID, "sop") for image in images]
            series_rows = [row for row in report if row["series_uid"] == clean_series]
            self.assertEqual([row["sop_uid"] for row in series_rows], expected)
            self.assertEqual([row["slice_index"] for row in series_rows], ["0", "1", "2"])
            self.assertEqual(len({row["series_uid"] for row in report}), 2)
            for index, entry in enumerate(series_rows):
                expected_mask, = (work / "data/inference_out/predictions").glob(f"*_slice{index:04d}_pred.png")
                np.testing.assert_array_equal(
                    cv2.imread(str(work / entry["prediction"]), cv2.IMREAD_GRAYSCALE),
                    cv2.imread(str(expected_mask), cv2.IMREAD_GRAYSCALE),
                )
            with sqlite3.connect(work / "output/metadata.db") as db:
                rows = db.execute("SELECT storage_key FROM dicom_metadata").fetchall()
            self.assertEqual({row[0] for row in rows}, {entry["storage_key"] for entry in report})

            for entry in report:
                local = work / "download.dcm"
                client.fget_object(bucket, entry["storage_key"], str(local))
                clean = pydicom.dcmread(local)
                self.assertFalse(clean.PatientName)
                self.assertNotEqual(clean.PatientID, "SYNTHETIC-123")
                self.assertEqual(clean.SOPInstanceUID, entry["sop_uid"])
                self.assertEqual(cv2.imread(str(work / entry["prediction"]), cv2.IMREAD_GRAYSCALE).shape, (32, 32))
                self.assertEqual(cv2.imread(str(work / entry["overlay"])).shape, (32, 32, 3))
                local.unlink()

            (raw / "c.dcm").rename(raw / "duplicate.dcm")
            images[0].save_as(raw / "c.dcm", enforce_file_format=True)
            cfg["paths"]["predictions"] = "rejected/predictions"
            (work / "config.yaml").write_text(yaml.safe_dump(cfg))
            log = run("dicom", "--etl-config", str(etl_cfg), expected_status=1)
            self.assertIn("Duplicate SOPInstanceUID", log)
            self.assertFalse((work / cfg["paths"]["predictions"]).exists())

            (raw / "duplicate.dcm").unlink()
            pdf = work / "data/PDFs/synthetic.pdf"
            pdf.parent.mkdir()
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Patient ID : SYNTHETIC-123")
                document.save(pdf)
            run("--config", str(etl_cfg), module="src.ETL.main")
            with sqlite3.connect(work / "output/metadata.db") as db:
                patient, key = db.execute("SELECT patient_id, storage_key FROM pdf_metadata").fetchone()
                patients = db.execute("SELECT DISTINCT patient_id FROM dicom_metadata").fetchall()
            self.assertEqual(patients, [(patient,)])
            client.fget_object(bucket, key, str(work / "sanitized.pdf"))
            with pymupdf.open(work / "sanitized.pdf") as document:
                self.assertNotIn("SYNTHETIC-123", document[0].get_text())
