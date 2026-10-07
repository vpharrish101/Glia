# Glia

Unified medical imaging pipeline that merges the ETL layer from **Task-1** with the ML segmentation pipeline from **Axion**.

## Project Structure

```
Glia/
├── config.yaml                 # ML pipeline config (preprocessing, training, inference)
├── config/
│   ├── config.yaml             # ETL pipeline config (input dirs, database path)
│   ├── deid.dicom              # DICOM de-identification rules
│   └── extra_cfg.dicom         # Additional DICOM header removal rules
├── src/
│   ├── main.py                 # Orchestrator: ETL → Train → Inference
│   ├── train.py                # UNet training with MLflow tracking
│   ├── inference.py            # Model inference and evaluation
│   ├── unet_arch.py            # UNet architecture definition
│   ├── ETL/
│   │   ├── main.py             # Standalone ETL entry point (DICOM + PDF)
│   │   ├── dicom_preprocess.py # DICOM de-identification and sanitization
│   │   ├── pdf_preprocess.py   # PDF field extraction and redaction
│   │   ├── Metadata_db.py      # SQLite metadata store
│   │   ├── Object_storage.py   # MinIO data lake client
│   │   ├── preprocessing.py    # NIfTI → 2.5D slice extraction for ML
│   │   ├── dataclass.py        # PyTorch Dataset + augmentations
│   │   └── utils/
│   │       ├── HMAC_builder.py # HMAC-based pseudonymisation
│   │       ├── load_config.py  # YAML config loader
│   │       └── logging_mod.py  # Structured logging setup
│   └── utils/
│       ├── metrics.py          # Dice / IoU metrics (torch + numpy)
│       └── visualize.py        # Overlay and comparison visualizations
├── data/
│   ├── DICOMs/                 # Raw DICOM input
│   ├── PDFs/                   # Raw PDF input
│   ├── raw/                    # Raw NIfTI subjects (BraTS format)
│   ├── processed/              # Extracted 2.5D slices + metadata
│   ├── model_chkpth/           # Saved model checkpoints
│   └── inference_out/          # Predictions and overlays
├── compose.yaml                # Docker Compose (MinIO + pipeline)
├── Dockerfile
├── requirements.txt            # Merged dependencies (ETL + ML)
├── .env.example
└── .gitignore
```

## Pipeline Stages

### 1. ETL — Healthcare Data Sanitization

The ETL layer from Task-1 provides:

- **DICOM processing**: De-identification via `deid`, HMAC pseudonymisation, deterministic UID regeneration, pixel PHI detection, and upload to MinIO.
- **PDF processing**: Field extraction via PyMuPDF, patient ID pseudonymisation, name redaction, and upload to MinIO.
- **Metadata storage**: SQLite-backed metadata for both DICOM and PDF records.

Run standalone:
```bash
python -m src.ETL.main
```

### 2. Preprocessing — NIfTI to 2.5D Slices

Converts raw NIfTI volumes (BraTS format) into 2.5D PNG slices with binary tumor masks, split into train/val sets.

### 3. Training — UNet Segmentation

Trains a UNet model with:
- Dice + BCE combined loss
- Mixed precision (AMP)
- MLflow experiment tracking
- ReduceLROnPlateau scheduler

### 4. Inference — Prediction and Evaluation

Loads the best checkpoint and runs inference on the validation set, computing Dice/IoU metrics and saving prediction overlays.

Run full pipeline:
```bash
python -m src.main
```

## Setup

1. Copy `.env.example` to `.env` and fill in secrets.
2. Place DICOM/PDF data in `data/DICOMs/` and `data/PDFs/`.
3. Place BraTS NIfTI data in `data/raw/`.
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python -m src.main`

Or via Docker:
```bash
docker compose up --build
```

## Environment Variables

| Variable             | Purpose                         |
|----------------------|---------------------------------|
| `HMAC_SECRET`        | HMAC key for pseudonymisation   |
| `MINIO_ENDPOINT`     | MinIO server address            |
| `MINIO_ROOT_USER`    | MinIO access key                |
| `MINIO_ROOT_PASSWORD`| MinIO secret key                |
| `MINIO_BUCKET`       | Target bucket name              |

## Origin

- **ETL layer** (`src/ETL/`): sourced from Task-1
- **ML pipeline** (`src/train.py`, `src/inference.py`, `src/unet_arch.py`, `src/utils/`): sourced from Axion
- **Glue** (`src/main.py`, configs, Dockerfile, compose): newly created for Glia
