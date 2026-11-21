# Glia

Glia combines DICOM ETL with Axion, a U-Net segmentation pipeline. Both commands start with DICOM images. Training also needs manual segmentation labels.

## Pipeline

- `src/ETL` sanitizes DICOMs, stores them in MinIO, records metadata in SQLite, and reads each series in physical slice order.
- `src/Axion` takes ordered image arrays, builds the neighboring-slice inputs, and trains or runs the segmentation model.
- `src/main.py` connects them and matches training masks to the input slices. The two packages are independent.

## Inputs

Put classic single-frame monochrome MR slices under `data/DICOMs/`. Subdirectories are supported. Use the sequence and orientation the model expects, such as FLAIR. Slices are grouped by patient, study, and series using DICOM metadata. By default, Axion uses the previous, current, and next slice from the same series.

For training, put one grayscale PNG mask per DICOM in `data/masks/`. Mirror the DICOM directory structure and append `.png` to its filename.

```text
data/DICOMs/patient-a/001.dcm
data/masks/patient-a/001.dcm.png
```

Each mask must match the original slice dimensions. Use zero for background and a positive value for foreground. Include blank masks for negative slices. Training needs at least two distinct patients so one can be held out for validation. Inference needs a checkpoint and DICOMs only.

The ETL removes identifying metadata and pseudonymizes patient IDs and UIDs. It requires `BurnedInAnnotation=NO` and rejects images flagged by the existing pixel rules. Verify that clinical inputs have no burned-in identifiers or recognizable anatomy before submitting them. The synthetic test checks the pipeline, and clinical accuracy requires representative labeled data.

## Run with Docker

Set `HMAC_SECRET`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, and `MINIO_BUCKET` in `.env`. Compose starts MinIO and will use the same storage settings for both commands.

```bash
docker compose run --rm --build pipeline train
docker compose run --rm pipeline inference
```

**Train** writes the best checkpoint to `data/model_chkpth/best.pth` and reports Dice and IoU on a patient-level validation split. **Inference** loads that checkpoint and writes masks, overlays, and `dicom_predictions.csv` under `data/inference_out/`. The CSV maps each PNG to its sanitized DICOM slice. Both commands save sanitized DICOMs in the configured bucket and metadata in `output/metadata.db`.

Both commands read the root `config.yaml` for input and output paths, ETL and data-loading workers, and model and training settings. Pass `--config` to use a different complete settings file. `training.learning_rate` is the optimizer update size, such as `0.0001`. `training.gradient_accumulation_steps` is the number of batches per update, such as `2`.

## Synthetic test

The test creates two synthetic patients with DICOM slices and PNG masks, trains one epoch, and runs inference. It checks that the stored DICOMs are sanitized and the output slices are in physical order.

```bash
docker compose run --rm --build --entrypoint python -e GLIA_TEST_S3=1 \
  -v "$PWD/tests:/app/tests:ro" pipeline -m unittest discover -s tests -v
```

The test uses a temporary bucket and cleans up its fixtures and checkpoint.

## Run from Python

Install the dependencies with Python 3.12 and `python -m pip install -r requirements.txt`. With an S3-compatible server running, set `MINIO_ENDPOINT` in `.env` to its host and port, such as `localhost:9010`.

```bash
python -m src.main train
python -m src.main inference
```

Some initial ETL and preprocessing code was developed using LLM assistance. The overall architecture, data flow, system design, and ML pipeline were designed and implemented by me.
