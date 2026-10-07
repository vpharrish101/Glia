import sqlite3

from src.ETL.utils.load_config import load_cfg
from src.ETL.utils.logging_mod import get_logger


logger=get_logger("MetadataStore")


class MetadataStore:
    def __init__(self):
        self.cfg=load_cfg()
        db_path=self.cfg["database"]["metadata_path"]

        from pathlib import Path
        Path(db_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sqlite=sqlite3.connect(db_path)

        logger.info(
            "Metadata store connected | path=%s",
            db_path,
        )

        self._init_DICOM()
        self._init_PDF()

    def _init_DICOM(self):
        self.sqlite.execute("""
            CREATE TABLE IF NOT EXISTS dicom_metadata (
                file_id TEXT PRIMARY KEY,
                patient_id TEXT,
                modality TEXT,

                study_uid TEXT,
                series_uid TEXT,
                sop_uid TEXT,

                manufacturer TEXT,
                manufacturer_model TEXT,
                body_part TEXT,

                rows INTEGER,
                columns INTEGER,
                pixel_spacing TEXT,
                slice_thickness TEXT,

                patient_sex TEXT,
                patient_age TEXT,

                pixel_status TEXT,
                processing_status TEXT,
                storage_key TEXT
            )
        """)

        self.sqlite.commit()

        logger.debug(
            "DICOM metadata table initialized"
        )

    def _init_PDF(self):
        self.sqlite.execute("""
            CREATE TABLE IF NOT EXISTS pdf_metadata (
                file_id TEXT PRIMARY KEY,
                patient_id TEXT,
                gender TEXT,
                age TEXT,
                gestational_age TEXT,
                bmi TEXT,
                storage_key TEXT
            )
        """)

        self.sqlite.commit()

        logger.debug(
            "PDF metadata table initialized"
        )

    def store_DICOM_metadata(self,deid_records):
        stored_count=0

        for deid_record in deid_records:
            if deid_record["status"]=="error":
                continue

            self.sqlite.execute("""
                INSERT OR REPLACE INTO dicom_metadata (
                    file_id,
                    patient_id,
                    modality,
                    study_uid,
                    series_uid,
                    sop_uid,
                    manufacturer,
                    manufacturer_model,
                    body_part,
                    rows,
                    columns,
                    pixel_spacing,
                    slice_thickness,
                    patient_sex,
                    patient_age,
                    pixel_status,
                    processing_status,
                    storage_key
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,(
                deid_record["filename"].removesuffix(".dcm"),
                deid_record["patient_id"],
                deid_record["modality"],
                deid_record["study_uid"],
                deid_record["series_uid"],
                deid_record["sop_uid"],
                deid_record["manufacturer"],
                deid_record["manufacturer_model"],
                deid_record["body_part"],
                deid_record["rows"],
                deid_record["columns"],
                deid_record["pixel_spacing"],
                deid_record["slice_thickness"],
                deid_record["patient_sex"],
                deid_record["patient_age"],
                deid_record["status"],
                deid_record["processing_status"],
                deid_record["storage_key"],
            ))

            stored_count+=1

        self.sqlite.commit()

        logger.info(
            "DICOM metadata committed | records=%d",
            stored_count,
        )

    def store_PDF_metadata(self,file_id,metadata,storage_key):
        self.sqlite.execute("""
            INSERT OR REPLACE INTO pdf_metadata (
                file_id,
                patient_id,
                gender,
                age,
                gestational_age,
                bmi,
                storage_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,(
            file_id,
            metadata["patient_id"],
            metadata["gender"],
            metadata["age"],
            metadata["gestational_age"],
            metadata["bmi"],
            storage_key,
        ))

        self.sqlite.commit()

        logger.info(
            "PDF metadata committed | file=%s",
            file_id,
        )

    def __enter__(self):
        logger.debug("Metadata store opened")
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        self.sqlite.close()
        logger.info("Metadata store connection closed")