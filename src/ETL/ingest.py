"""Sanitize DICOM inputs and persist the resulting images and metadata."""

import os
from pathlib import Path

from src.ETL.Metadata_db import MetadataStore
from src.ETL.Object_storage import DataLake
from src.ETL.dicom_preprocess import processDICOMBatch


def ingest_dicoms(source_dir, metadata_path, num_workers=4):
    if not os.getenv("HMAC_SECRET"):
        raise ValueError("HMAC_SECRET must be set for DICOM ingestion")
    if not Path(source_dir).is_dir():
        raise ValueError("The configured DICOM input directory does not exist")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("etl.num_workers must be a non-negative integer")

    lake = DataLake()
    records = processDICOMBatch(source_dir, lake, num_workers)
    if not records:
        raise ValueError("No .dcm or .dicom files found in the DICOM input directory")
    with MetadataStore(metadata_path) as store:
        store.store_DICOM_metadata(records)
    if any(record["status"] == "error" for record in records):
        raise ValueError("DICOM ingestion rejected one or more inputs. See the ingestion log.")
    return lake, records
