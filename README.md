# Glia

**What it is:** An end-to-end medical imaging pipeline combining a healthcare ETL system with a medical image segmentation pipeline.

**What it does:** Glia takes medical imaging data through two main stages: first, it processes and de-identifies DICOM/PDF data and stores the resulting files and metadata; then, it prepares medical images for training, trains a U-Net segmentation model, and evaluates its predictions.

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


The ML pipeline is therefore concerned with the prepared imaging data rather than the original patient records.

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

## Deployment: -

### With Docker

```bash
docker compose up --build
```

### Manual Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```


