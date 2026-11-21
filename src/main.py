"""Connect DICOM ETL to Axion training and inference."""

import argparse
import csv
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("Glia")


def main():
    parser = argparse.ArgumentParser(description="Glia DICOM segmentation")
    parser.add_argument("mode", choices=("train", "inference"))
    parser.add_argument("--config", default="config.yaml", help="Axion configuration")
    parser.add_argument("--etl-config", help="Override DICOM and SQLite paths with a legacy ETL configuration")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    dicom_dir = paths.get("dicom", "data/DICOMs")
    metadata_path = paths.get("metadata_db", "output/metadata.db")
    if args.etl_config:
        etl_cfg = yaml.safe_load(Path(args.etl_config).read_text())
        dicom_dir = etl_cfg["Input_dirs"]["DICOM_dir"]
        metadata_path = etl_cfg["database"]["metadata_path"]
    logging.basicConfig(
        level=cfg.get("logging", {}).get("level", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.Axion.dataset import prepare_scan
    from src.ETL.ingest import ingest_dicoms
    from src.ETL.series import read_series

    training = args.mode == "train"
    mask_dir = Path(cfg["paths"].get("masks", "data/masks"))
    manifest_rows = {}

    def scans():
        lake, records = ingest_dicoms(dicom_dir, metadata_path, cfg.get("etl", {}).get("num_workers", 4))
        if any(record["modality"] != "MR" for record in records):
            raise ValueError("Axion expects MR series compatible with the model")

        for images, ordered in read_series(lake, records):
            slice_ids = [record["sop_uid"] for record in ordered]
            mask_paths = None
            if training:
                mask_paths = [mask_dir / (record["source_path"].as_posix() + ".png") for record in ordered]
            else:
                for index, record in enumerate(ordered):
                    manifest_rows[record["sop_uid"]] = (
                        record["study_uid"], record["series_uid"], record["sop_uid"], index, record["storage_key"],
                    )
            yield prepare_scan(ordered[0]["patient_id"], slice_ids, images, cfg["preprocessing"]["image_size"], mask_paths)

    try:
        if training:
            from src.Axion.train import train

            if not mask_dir.is_dir():
                raise ValueError(f"Training requires a PNG mask for each DICOM slice under {mask_dir}")
            train(cfg, scans())
        else:
            from src.Axion.inference import run_inference

            predictions = run_inference(cfg, scans())
            manifest = Path(cfg["paths"]["predictions"]) / "dicom_predictions.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("study_uid", "series_uid", "sop_uid", "slice_index", "storage_key", "prediction", "overlay"))
                writer.writerows((*manifest_rows[uid], *paths) for uid, paths in predictions.items())
            logger.info("Predicted %d DICOM slices. Manifest: %s", len(predictions), manifest)
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
