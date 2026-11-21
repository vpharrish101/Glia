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
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.ETL.series import order_slices
from src.ETL.dicom_preprocess import RejectedDicom, sanitize_DICOM
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
    def test_train_on_dicom_and_predict_sanitized_series(self):
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

            cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
            cfg["preprocessing"]["image_size"] = 32
            cfg["training"].update(epochs=1, batch_size=16, learning_rate=0.0001, num_workers=0)
            cfg["etl"]["num_workers"] = 2
            cfg["model"]["features"] = [4, 8]
            cfg["inference"]["checkpoint"] = "checkpoints/synthetic.pth"
            cfg["paths"].update(
                dicom="incoming/dicom", masks="annotations", metadata_db="metadata/index.db",
                predictions="dicom/predictions", overlays="dicom/overlays",
            )
            config = work / "config.yaml"
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
                label = labels / (filename + ".png")
                label.parent.mkdir(parents=True, exist_ok=True)
                self.assertTrue(cv2.imwrite(str(label), phantom(position)[1] * 255))

            training = run("train")
            self.assertIn("V-Dice", training)
            self.assertIn("LR 0.00010", training)
            self.assertRegex(training, r"Training on 1 patients \([34] slices\), validating on 1 patients")
            checkpoint = work / cfg["inference"]["checkpoint"]
            self.assertGreater(checkpoint.stat().st_size, 0)
            with sqlite3.connect(work / cfg["paths"]["metadata_db"]) as db:
                rows = db.execute("SELECT storage_key FROM dicom_metadata").fetchall()
            self.assertEqual(len(rows), 7)

            # Inference reads the same input layout without labels.
            labels.rename(work / "unused-labels")
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
