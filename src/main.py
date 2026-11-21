import argparse
import logging
from pathlib import Path

logger = logging.getLogger("Glia.Orchestrator")


def _run_dicom(config_path, etl_config_path):
    import os

    import torch
    from dotenv import load_dotenv

    from src.ETL.Metadata_db import MetadataStore
    from src.ETL.Object_storage import DataLake
    from src.ETL.main import ingest_dicoms
    from src.ETL.preprocessing import context_slices
    from src.ETL.utils.load_config import load_cfg
    from src.inference import load_config, load_model, run_dicom_inference

    load_dotenv()
    if not os.getenv("HMAC_SECRET"):
        raise ValueError("HMAC_SECRET must be set for DICOM ingestion")
    cfg = load_config(config_path)
    etl_cfg = load_cfg(etl_config_path)
    if not Path(etl_cfg["Input_dirs"]["DICOM_dir"]).is_dir():
        raise ValueError("The configured DICOM input directory does not exist")
    context_slices(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["inference"]["checkpoint"], cfg, device)

    lake = DataLake()
    with MetadataStore(etl_cfg["database"]["metadata_path"]) as store:
        records = ingest_dicoms(etl_cfg, store, lake)
    if not records:
        raise ValueError("No .dcm or .dicom files found in the DICOM input directory")
    if any(record["status"] == "error" for record in records):
        raise ValueError("DICOM ingestion rejected one or more inputs. See the ingestion log before retrying.")
    if any(record["modality"] != "MR" for record in records):
        raise ValueError("Segmentation expects MR series compatible with the checkpoint. Use clinical ETL for CT.")
    run_dicom_inference(cfg, model, device, lake, records)


def main():
    parser = argparse.ArgumentParser(description="Glia segmentation pipeline")
    parser.add_argument("mode", choices=("train", "dicom"), nargs="?", default="train")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--etl-config", default="config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        if args.mode == "dicom":
            _run_dicom(args.config, args.etl_config)
        else:
            from src.ETL.preprocessing import run_etl
            from src.inference import run_inference
            from src.train import train

            run_etl(args.config)
            train(args.config)
            run_inference(args.config)
    except Exception:
        logger.exception("Pipeline failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
