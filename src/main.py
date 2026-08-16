from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from collections import defaultdict

from src.Metadata_db import MetadataStore
from src.dicom_preprocess import process_DICOM_batch
from src.utils import load_config
from src.pdf_preprocess import sanitize_PDF
from src.Object_storage import DataLake
from src.utils.logging_mod import get_logger


logger=get_logger("Pipeline")
cfg=load_config.load_cfg()


def _DICOM(meta_store,lake):
    logger.info("Starting DICOM pipeline")

    deid_records=process_DICOM_batch(
        cfg["Input_dirs"]["DICOM_dir"],
        lake,)

    meta_store.store_DICOM_metadata(deid_records)

    processed_count=sum(1 for record in deid_records if record["status"]!="error")

    failed_count=sum(
        1
        for record in deid_records
        if record["status"]=="error"
    )

    logger.info(
        "DICOM pipeline completed | processed=%d | failed=%d",
        processed_count,
        failed_count,
    )


def _PDF(meta_store,lake):
    PDF_source_dir=Path(
        cfg["Input_dirs"]["PDF_dir"]
    )

    patient_sequences=defaultdict(int)

    PDF_count=0
    PDF_skipped=0
    PDF_failed=0

    logger.info(
        "Starting PDF pipeline | source=%s",
        PDF_source_dir,
    )

    for PDF_file in sorted(PDF_source_dir.glob("*.pdf")):

        PDF_count+=1
        logger.info("Processing PDF | file=%s",PDF_file.name,)

        try:
            PDF_metadata=sanitize_PDF(
                PDF_file,
                lake,
                patient_sequences,)

            if PDF_metadata is None:
                PDF_skipped+=1

                logger.warning("Skipped PDF | file=%s | reason=missing Patient ID",PDF_file.name,)

                continue

            meta_store.store_PDF_metadata(
                file_id=PDF_metadata["filename"].removesuffix(
                    ".pdf"
                ),
                metadata=PDF_metadata,
                storage_key=PDF_metadata["storage_key"],
            )

            logger.info(
                "PDF completed | file=%s | object=%s",
                PDF_file.name,
                PDF_metadata["storage_key"],
            )

        except Exception:
            PDF_failed+=1

            logger.exception(
                "PDF processing failed | file=%s",
                PDF_file.name,
            )

    processed_count=PDF_count-PDF_skipped-PDF_failed

    logger.info(
        "PDF pipeline completed | total=%d | processed=%d | skipped=%d | failed=%d",
        PDF_count,
        processed_count,
        PDF_skipped,
        PDF_failed,
    )


def main():
    logger.info("ETL pipeline started")
    lake=DataLake()

    with MetadataStore() as meta_store:
        _DICOM(meta_store,lake,)
        #_PDF(meta_store,lake,)

    logger.info("ETL pipeline completed successfully")

if __name__=="__main__":
    main()