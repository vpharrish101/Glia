# Glia

**What it is:** An end-to-end medical imaging pipeline combining a healthcare ETL system with a medical image segmentation pipeline.

**What it does:** Glia ingests DICOM/PDF data into object storage and records metadata in SQLite. It trains and evaluates a U-Net on labeled NIfTI volumes. For DICOM inference, Glia ingests raw MR slices, retrieves the sanitized images from object storage, and runs a trained checkpoint on them.

**What's unique:** The project connects the data-processing side and the ML side instead of treating them as two separate projects. The ETL layer deals with the clinical data itself, while the ML layer works on the resulting medical imaging data.


## Summary: -

Glia is essentially two pipelines joined together.

The first part deals with the ETL of medical data:

<img width="1444" height="220" alt="image" src="https://github.com/user-attachments/assets/03c91fb3-30fe-49d1-81d0-bf5ad669f05a" />


DICOM files are sanitized and patient identifiers are pseudonymised using HMAC. Relevant metadata is stored separately in SQLite, while the actual medical files are kept in MinIO. PDF reports are processed as part of the same ETL layer.

The second part deals with the machine learning workflow:

```text
Medical Imaging
      ↓
Preprocessing
      ↓
2.5D Dataset
      ↓
U-Net Training
      ↓
Inference
      ↓
Dice / IoU
```

The preprocessing stage turns the medical imaging data into training samples. The model then performs binary segmentation, with MLflow used to keep track of training runs and results. Overall workflow is discussed in detail, later in the section. Some initial ETL and preprocessing code was developed using LLM assistance. The overall architecture, data flow, system design, and ML pipeline were designed and implemented by me.

  

## Why it matters: -

A medical imaging model needs more than an image folder. The data has to be processed, patient information has to be handled properly, metadata needs to stay associated with the right studies, and the resulting data has to be converted into a form the model can use.

Glia puts those steps into one workflow.

The ETL layer is responsible for getting the medical data into a usable state. The ML layer then works on that prepared data without having to deal with the original clinical files directly. This also means that the storage and data-processing side can change independently from the segmentation model.



---

## Architecture: -

<img width="1321" height="510" alt="image" src="https://github.com/user-attachments/assets/00e0361c-9adc-4e3d-9a67-340b8c523c58" />



The architecture is split into two main sections.

**ETL Pipeline: -**


<img width="946" height="418" alt="image" src="https://github.com/user-attachments/assets/6e8bdcc7-e189-4264-862b-bb04d13f7dd7" />

MinIO stores the medical imaging objects, while SQLite stores the metadata needed to work with them. Keeping these separate avoids putting large medical files directly into the metadata database.


**ML Pipeline: -**

<img width="1456" height="287" alt="image" src="https://github.com/user-attachments/assets/18a2345f-fd25-44a6-9427-7d385f72ec0e" />


DICOM inference builds the same neighboring-slice representation used for NIfTI training. It orders stored slices by patient-space position within each series and keeps patient identifiers out of prediction filenames.

---

## Structure Dynamics: -

The reason I split the project this way: -
```
1. The ETL layer handles DICOM and PDF files before they enter the ML workflow. This includes removing identifying information,
   pseudonymising patients, handling DICOM UIDs, and extracting metadata.

2. MinIO and SQLite have different jobs. MinIO holds the actual medical files, while SQLite keeps the structured information
   about those files.

3. The preprocessing stage converts medical imaging into samples that can actually be consumed by the segmentation model. The
   project uses a 2.5D representation, where neighboring slices provide additional context to the model.

4. The U-Net handles the actual segmentation task. It is trained on the prepared dataset rather than directly interacting with
   the healthcare ETL layer.

5. MLflow sits around the training process rather than inside the model itself. It records experiments, metrics and model checkpoints
   so different training runs can be compared.

6. The stages are kept separate so that changing the segmentation model does not require rebuilding the DICOM processing system, and
   changes to the data pipeline do not require rewriting the training code.
```
---

## Features: -

- **Healthcare ETL** - DICOM/PDF ingestion, de-identification, patient pseudonymisation, UID handling, and metadata extraction.
- **Medical Data Storage** - MinIO for imaging files and SQLite for associated metadata.
- **ML Dataset Construction** - Medical image preprocessing, 2.5D sample generation, and subject-level train/validation splitting.
- **Segmentation Pipeline** - U-Net training, inference, checkpointing, and Dice / IoU evaluation.
- **Experiment Tracking** - MLflow for training runs, metrics, and model artifacts.
- **Containerized Workflow** - Docker Compose for the pipeline environment and supporting services.
  
## MLOps Design: -

- **Separate stages** - ETL, preprocessing, training and inference are separate parts of the pipeline.
- **Configuration-driven** - paths and pipeline parameters are kept outside the main processing logic.
- **Reusable preprocessing** - the processed dataset can be reused across multiple training runs.
- **Experiment tracking** - MLflow keeps training runs and results together.
- **Dockerized services** - MinIO and the pipeline environment can be started through Docker Compose.

The main reason for this structure is practical: preprocessing and data handling should not have to be repeated every time a model parameter changes.

---

## Installation

Use Python 3.12 from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CPU-only Linux x86-64, pass `--extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.7.0+cpu' 'torchvision==0.22.0+cpu'` to the install command. The Docker image uses these CPU wheels.

## Train on labeled NIfTI

Put at least two subject directories under `data/raw`, with a FLAIR image and a corresponding segmentation volume on the same voxel grid in each directory. Filenames must contain `flair` and `seg` respectively. Configure paths and settings in `config.yaml`.

```bash
python -m src.main train --config config.yaml
```

Training writes the best checkpoint to `inference.checkpoint`, runs validation inference, and reports Dice and IoU. Running `python -m src.main` also selects this workflow.

## Run DICOM inference

1. Set `HMAC_SECRET`, `MINIO_ENDPOINT`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, and `MINIO_BUCKET` in the environment or a local `.env`. `MINIO_ENDPOINT` is the host and port of a reachable S3-compatible server. Keep the HMAC secret stable across runs that need to link the same studies.
2. Put `.dcm` or `.dicom` files under `Input_dirs.DICOM_dir` in `config/config.yaml` (`data/DICOMs` by default). Submit classic, single-frame monochrome MR series compatible with the checkpoint. Multi-slice series need consistent geometry and unique, regularly spaced slice positions. Provide only the sequences the model expects, such as the FLAIR series for a FLAIR model.
3. Set `inference.checkpoint` to an existing compatible U-Net checkpoint in `config.yaml`. The model configuration, native image orientation, slice order, and preprocessing must match the assumptions used to train that checkpoint. This workflow uses the source image axes and performs no atlas registration.

```bash
python -m src.main dicom --config config.yaml --etl-config config/config.yaml
```

The DICOM command ingests the current batch, records accepted objects in SQLite, and reads those sanitized objects back for inference. A rejected input stops prediction for the batch. Accepted objects can still be present in storage. Each series is handled independently. Under `paths.predictions` and `paths.overlays`, the command writes PNGs on the model's resized grid and a `dicom_predictions.csv` manifest that maps them to pseudonymous study, series, and SOP instance UIDs. DICOM inference needs no segmentation labels. Dice and IoU require labeled validation data.

The DICOM metadata allowlist removes free text and private elements and pseudonymizes identifiers. The pixel policy requires `BurnedInAnnotation=NO` and rejects images flagged by the project rules. It does not detect all burned-in text or recognizable anatomy. Verify those before submitting clinical data. The synthetic tests verify pipeline behavior and do not establish clinical de-identification or model accuracy.

To ingest DICOMs and PDFs without segmentation, run `python -m src.ETL.main --config config/config.yaml`. Standalone DICOM ingestion also accepts classic single-frame monochrome CT.

### Docker Compose

Set the secret and storage variables in `.env`, place the DICOM inputs and model checkpoint under the mounted `data/` directory, then run:

```bash
docker compose up --build
```

Compose runs DICOM inference using its MinIO service. If a trusted network proxy intercepts package downloads during a standalone image build, provide the trusted CA bundle using `docker build --secret id=pip_ca,src=/path/to/ca-bundle.pem -t glia .`.

### Synthetic integration test

With a local test S3-compatible server and its `MINIO_ENDPOINT`, `MINIO_ROOT_USER`, and `MINIO_ROOT_PASSWORD` exported, run:

```bash
GLIA_TEST_S3=1 python -m unittest discover -s tests -v
```

The test uses a temporary bucket. It trains one epoch on labeled NIfTI phantoms, encodes equivalent image arrays as DICOM, and checks that predictions agree after sanitization, storage, and retrieval.
