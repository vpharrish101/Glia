import pydicom
import tempfile
import src.utils.HMAC_builder as HMAC_builder

from pydicom.uid import generate_uid
from pathlib import Path
from collections import defaultdict
from deid.config import DeidRecipe
from deid.dicom import DicomCleaner
from deid.dicom import get_identifiers
from deid.dicom import replace_identifiers

from src.utils.logging_mod import get_logger


logger=get_logger("DICOM")


def sanitize_DICOM(source_file,
                   uid_registry,
                   lake,
                   object_name,):

    source_file=Path(source_file)
    logger.info("Sanitizing DICOM file: %s",source_file.name,)

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_dir=Path(tmp_dir)
        original=pydicom.dcmread(source_file,stop_before_pixels=True,)

        logger.debug("Loaded DICOM metadata: %s",source_file.name,)

        deid_recipe=DeidRecipe(
            deid=[
                "config/extra_cfg.dicom",
                "config/deid.dicom",
            ]
        )

        identifiers=get_identifiers(
            [str(source_file)],
            expand_sequences=False,
            )


        for fields in identifiers.values():
            patient_id=next(
                (
                    str(x.element.value)
                    for x in fields.values()
                    if x.name=="PatientID"
                ),
                None,
            )

            if patient_id:
                fields["entity_id"]=HMAC_builder.patient_hmac(
                    patient_id
                )

        logger.debug(
            "Applying DICOM de-identification rules: %s",
            source_file.name,
        )

        replace_identifiers(
            dicom_files=[str(source_file)],
            ids=identifiers,
            deid=deid_recipe,
            save=True,
            output_folder=str(tmp_dir),
            remove_private=True,
            strip_sequences=False,
        )

        deid_file=tmp_dir/source_file.name
        DICOM=pydicom.dcmread(deid_file)

        
        if hasattr(original, "StudyInstanceUID"):
            DICOM.StudyInstanceUID = HMAC_builder.deterministic_uid(
                str(original.StudyInstanceUID),
                "study",
            )

        if hasattr(original, "SeriesInstanceUID"):
            DICOM.SeriesInstanceUID = HMAC_builder.deterministic_uid(
                str(original.SeriesInstanceUID),
                "series",
            )

        if hasattr(original, "SOPInstanceUID"):
            new_sop_uid = HMAC_builder.deterministic_uid(
                str(original.SOPInstanceUID),
                "sop",
            )

            DICOM.SOPInstanceUID = new_sop_uid

            if hasattr(DICOM, "file_meta"):
                DICOM.file_meta.MediaStorageSOPInstanceUID = new_sop_uid

        if hasattr(original, "FrameOfReferenceUID"):
            DICOM.FrameOfReferenceUID = HMAC_builder.deterministic_uid(
                str(original.FrameOfReferenceUID),
                "frame",
            )

        DICOM.save_as(deid_file)

        logger.debug("DICOM UIDs regenerated: %s",source_file.name,)
        pixel_cleaner=DicomCleaner(deid="config/deid.dicom")
        pixel_scan=pixel_cleaner.detect(str(deid_file))
        pixel_status="gtg"
        pixel_reason="no pixel filter matched"

        if pixel_scan["flagged"]:
            pixel_results=pixel_scan["results"]

            cleanable_pixels=[
                item
                for item in pixel_results
                if item["group"]=="graylist"
                and item["coordinates"]
            ]

            whitelist_hits=[
                item
                for item in pixel_results
                if item["group"]=="whitelist"
            ]

            blacklist_hits=[
                item
                for item in pixel_results
                if item["group"]=="blacklist"
            ]

            graylist_hits=[
                item
                for item in pixel_results
                if item["group"]=="graylist"
            ]

            if cleanable_pixels:
                pixel_cleaner.clean()

                pixel_cleaner.save_dicom(output_folder=str(tmp_dir))

                cleaned_file=(tmp_dir/f"cleaned-{deid_file.name}")

                if cleaned_file.exists():
                    deid_file.unlink()
                    cleaned_file.rename(deid_file)

                pixel_status="graylist_cleaned"
                pixel_reason="; ".join(
                    item["reason"]
                    for item in cleanable_pixels
                )

                logger.warning("Pixel data cleaned for DICOM: %s",source_file.name,)

            elif whitelist_hits:
                pixel_status="whitelist"
                pixel_reason="; ".join(
                    item["reason"]
                    for item in whitelist_hits
                )

                logger.info(
                    "Pixel rule matched whitelist: %s",
                    source_file.name,
                )

            elif blacklist_hits:
                pixel_status="blacklist"
                pixel_reason="; ".join(
                    item["reason"]
                    for item in blacklist_hits
                )

                logger.error(
                    "Pixel rule matched blacklist: %s",
                    source_file.name,
                )

            elif graylist_hits:
                pixel_status="graylist_review"
                pixel_reason="; ".join(
                    item["reason"]
                    for item in graylist_hits
                )

                logger.warning(
                    "DICOM requires pixel review: %s",
                    source_file.name,
                )

        lake_key=lake.put(deid_file,f"dicom/{object_name}",)
        logger.info("DICOM stored in data lake: %s",lake_key,)

        return {
            "filename":object_name,
            "patient_id":str(DICOM.PatientID),
            "status":pixel_status,
            "reason":pixel_reason,
            "storage_key":lake_key,
            "modality":getattr(DICOM,"Modality",None),
            "study_uid":getattr(DICOM,"StudyInstanceUID",None),
            "series_uid":getattr(DICOM,"SeriesInstanceUID",None),
            "sop_uid":getattr(DICOM,"SOPInstanceUID",None),
            "manufacturer":getattr(DICOM,"Manufacturer",None),
            "manufacturer_model":getattr(
                DICOM,
                "ManufacturerModelName",
                None,
            ),
            "body_part":getattr(DICOM,"BodyPartExamined",None),
            "rows":getattr(DICOM,"Rows",None),
            "columns":getattr(DICOM,"Columns",None),
            "pixel_spacing":str(
                getattr(
                    DICOM,
                    "PixelSpacing",
                    None,
                )
            ),
            "slice_thickness":str(
                getattr(
                    DICOM,
                    "SliceThickness",
                    None,
                )
            ),
            "patient_sex":getattr(DICOM,"PatientSex",None),
            "patient_age":getattr(DICOM,"PatientAge",None),
            "processing_status":"success",
        }


def process_DICOM_batch(source_dir,lake):
    source_dir=Path(source_dir)
    logger.info("Starting DICOM batch: %s",source_dir,)

    uid_registry={
        "study":{},
        "series":{},
        "frame":{},
    }

    patient_sequences=defaultdict(int)
    deid_records=[]

    DICOM_files=sorted(source_dir.rglob("*.dcm"))

    logger.info("DICOM files discovered: %d",len(DICOM_files),)

    for source_file in DICOM_files:
        try:
            DICOM=pydicom.dcmread(
                source_file,
                stop_before_pixels=True,
            )

            original_patient_id=getattr(DICOM,"PatientID",None,)

            if original_patient_id is None:
                raise ValueError("Missing PatientID")

            patient_id=HMAC_builder.patient_hmac(str(original_patient_id))
            original_sop_uid = str(DICOM.SOPInstanceUID)

            object_name = (
                f"{patient_id}_"
                f"{HMAC_builder.deterministic_uid(original_sop_uid, 'sop')}.dcm"
            )

            deid_record=sanitize_DICOM(
                source_file,
                uid_registry,
                lake,
                object_name,
            )

            deid_records.append(deid_record)

        except Exception as error:
            logger.exception(
                "DICOM processing failed: %s",
                source_file.name,
            )

            deid_records.append({
                "filename":"",
                "patient_id":"",
                "status":"error",
                "reason":str(error),
            })

    logger.info("DICOM batch completed: processed=%d",len(deid_records),)

    return deid_records