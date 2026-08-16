# Healthcare Dataset ETL Pipeline (Task-1): -

## 1. Architecture Overview

The pipeline is designed as a batch preprocessing pipeline for healthcare data consisting of DICOM images and PDF reports.

The implementation is split into two preprocessing paths:

<img width="946" height="418" alt="image" src="https://github.com/user-attachments/assets/6b9e42e6-4825-455a-aa42-7465cdd6623f" />

### Psuedocode: -
```
load config + environment variables
connect to metadata store + object storage

for each DICOM file:
    read metadata
    de-identify + pseudonymise patient ID
    regenerate UIDs
    check pixel data for PHI
    clean / flag if required
    upload sanitized DICOM
    store metadata

for each PDF file:
    extract patient fields
    pseudonymise Patient ID
    redact Patient ID + Patient Name
    clear PDF metadata
    upload sanitized PDF
    store metadata

catch file errors -> log -> continue batch
```
### DICOM path

DICOM files are read with `pydicom` and processed using the `deid` package.

The pipeline:
- applies the configured DICOM de-identification rules
- removes private information
- pseudonymises the patient identifier using HMAC
- regenerates Study, Series, SOP and Frame of Reference UIDs deterministically
- checks pixel data for possible PHI
- uploads the sanitized DICOM to the data lake
- stores the resulting metadata

### PDF path

PDF reports are processed using PyMuPDF.

The pipeline extracts the configured fields:

- Patient ID
- Patient Name
- Gender
- Patient Age
- GA
- BMI

Patient ID is pseudonymised using the same HMAC mechanism, while Patient Name is replaced with `REDACTED`. The located Patient ID and Patient Name are also redacted from the PDF itself before storage.


### Storage

Sanitized files are stored through the `DataLake` class using MinIO. Metadata is stored separately using SQLite through the `MetadataStore` class.

---

## 2. Setup and Execution Instructions

1. Clone the repository and open the project from its root directory.
2. Make sure the required environment variables are available and set:
   - `HMAC_SECRET`
   - `MINIO_ENDPOINT`
   - `MINIO_ROOT_USER`
   - `MINIO_ROOT_PASSWORD`
   - `MINIO_BUCKET`
3. Add DICOM data in **data/DICOMs/** and PDF at **data/PDFs/**
4. Configure the input directories and metadata database path in  ```text config/config.yaml ```
5. Run ```text docker compose up --build```

   
## 3. Configuration Explanation

Pipeline paths and database configuration are loaded from: ```text config/config.yaml```

The HMAC secret is supplied through: ```text HMAC_SECRET ```

The HMAC secret is used for:
- patient pseudonymisation
- deterministic DICOM UID generation

The MinIO client reads its connection details from environment variables:

```text
MINIO_ENDPOINT
MINIO_ROOT_USER
MINIO_ROOT_PASSWORD
MINIO_BUCKET
```

The metadata database path is loaded from the database configuration in `config/config.yaml`.

## 4. Design Decisions and Trade-offs

- **SQLite for metadata:** Simple and lightweight for the current workload, but concurrent writes and horizontal scaling are limited.

- **MinIO for file storage:** Keeps DICOM/PDF objects out of the database and makes storage easier to manage. The trade-off is keeping metadata and object storage in sync.

- **Deterministic HMAC IDs:** Keeps patient identifiers stable across reruns without exposing the original ID. The trade-off is that the same secret must be protected since it controls the mapping.

- **Per-file failure handling:** A failed file does not stop the complete batch. The error is logged and processing continues with the remaining files.

- **Rule-based PDF extraction:** Kept the initial implementation simple and predictable for the expected report format. The trade-off is that arbitrary PDF layouts will not generalise as well.
