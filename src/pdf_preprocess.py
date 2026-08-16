import pymupdf as fitz
import tempfile


from pathlib import Path
from src.utils import HMAC_builder
from src.utils.logging_mod import get_logger


logger=get_logger("PDF")


PDF_FIELDS={
    "Patient ID :":"patient_id",
    "Patient Name :":"patient_name",
    "Gender :":"gender",
    "Patient Age :":"age",
    "GA :":"gestational_age",
    "BMI :":"bmi",
}


def sanitize_PDF(source_file,
                 lake,
                 patient_sequences,):

    source_file=Path(source_file)
    logger.info("Sanitizing PDF | file=%s",source_file.name,)

    PDF_metadata={
        field_name:None
        for field_name in PDF_FIELDS.values()
    }

    with tempfile.TemporaryDirectory() as tmp_dir:

        PDF=fitz.open(source_file)
        logger.debug(
            "PDF opened | file=%s | pages=%d",
            source_file.name,
            len(PDF),
        )

        for page in PDF:
            words=page.get_text("words")
            for label,field_name in PDF_FIELDS.items():
                label_rects=page.search_for(label)
                if not label_rects:
                    continue

                label_rect=label_rects[0]
                next_x=page.rect.width
                for other_label in PDF_FIELDS:
                    if other_label==label:
                        continue
                    other_rects=page.search_for(other_label)

                    if not other_rects:
                        continue

                    other_rect=other_rects[0]
                    same_row=(abs(other_rect.y0-label_rect.y0)<5)
                    to_right=(other_rect.x0>label_rect.x1)

                    if same_row and to_right:
                        next_x=min(next_x,other_rect.x0,)

                field_words=[]


                for word in words:
                    x0,y0,x1,y1,text=word[:5]
                    same_row=(abs(y0-label_rect.y0)<5)
                    after_label=(x0>=label_rect.x1)
                    before_next=(x0<next_x)

                    if (
                        same_row
                        and after_label
                        and before_next
                    ):
                        field_words.append(word)

                field_words.sort(key=lambda word:word[0])
                field_value=" ".join(
                        word[4]
                        for word in field_words
                    ).strip()

                if not field_value:
                    continue

                if field_name=="patient_id":
                    sanitized_value=(
                        HMAC_builder.patient_hmac(
                            field_value
                        )
                    )

                elif field_name=="patient_name":
                    sanitized_value="REDACTED"

                else:
                    sanitized_value=field_value

                PDF_metadata[field_name]=sanitized_value

                if field_name in (
                    "patient_id",
                    "patient_name",
                ):
                    redact_rect=fitz.Rect(
                        field_words[0][0],
                        min(word[1]
                            for word in field_words
                        ),
                        field_words[-1][2],
                        max(
                            word[3]
                            for word in field_words
                        ),
                    )

                    page.add_redact_annot(
                        redact_rect,
                        text=sanitized_value,
                    )

            page.apply_redactions()

        if not PDF_metadata["patient_id"]:
            PDF.close()

            logger.warning(
                "PDF skipped | file=%s | reason=missing Patient ID",
                source_file.name,
            )

            return None

        patient_id=PDF_metadata["patient_id"]

        patient_sequences[patient_id]+=1

        object_name=(
            f"{patient_id}_"
            f"{patient_sequences[patient_id]:03d}.pdf"
        )

        tmp_pdf=Path(tmp_dir)/object_name

        PDF.set_metadata({})
        PDF.save(tmp_pdf)
        PDF.close()

        logger.debug(
            "Sanitized PDF written to temporary path | file=%s",
            source_file.name,
        )

        lake_key=lake.put(
            tmp_pdf,
            f"pdf/{object_name}",
        )

        logger.info(
            "PDF stored in data lake | file=%s | key=%s",
            source_file.name,
            lake_key,
        )

        PDF_metadata["storage_key"]=lake_key
        PDF_metadata["filename"]=object_name

        return PDF_metadata