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
    parser.add_argument("--config", default="config/config.yaml", help="Pipeline configuration")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]
    logging.basicConfig(
        level=cfg.get("logging", {}).get("level", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.Axion.dataset import prepare_scan
    from src.ETL.ingest import ingest_dicoms
    from src.ETL.segmentation import read_segmentation_masks
    from src.ETL.series import read_series

    training = args.mode == "train"
    manifest_rows = {}

    def scans():
        segmentation_paths = [] if training else None
        lake, records = ingest_dicoms(
            paths["dicom"], paths["metadata_db"], cfg["etl"]["num_workers"], segmentation_paths=segmentation_paths,
        )
        if any(record["modality"] != "MR" for record in records):
            raise ValueError("Axion expects MR series compatible with the model")

        seg_masks = None
        mask_dir = None
        if segmentation_paths:
            seg_masks = read_segmentation_masks(segmentation_paths, records, cfg["training"].get("segment_number"))
            unlabeled = {record["sop_uid"] for record in records} - seg_masks.keys()
            if unlabeled:
                raise ValueError(f"DICOM SEG does not label {len(unlabeled)} input slices")
        elif training:
            mask_dir = Path(paths["masks"])
            if not mask_dir.is_dir():
                raise ValueError(f"Training requires DICOM SEG or a PNG mask for each slice under {mask_dir}")

        for images, ordered in read_series(lake, records):
            slice_ids = [record["sop_uid"] for record in ordered]
            mask_paths = None
            mask_pixels = None
            if seg_masks is not None:
                mask_pixels = [seg_masks.pop(uid) for uid in slice_ids]
            elif mask_dir is not None:
                mask_paths = [mask_dir / (record["source_path"].as_posix() + ".png") for record in ordered]
            else:
                for index, record in enumerate(ordered):
                    manifest_rows[record["sop_uid"]] = (
                        record["study_uid"], record["series_uid"], record["sop_uid"], index, record["storage_key"],
                    )
            yield prepare_scan(
                ordered[0]["patient_id"], slice_ids, images, cfg["preprocessing"]["image_size"],
                mask_paths, mask_pixels=mask_pixels,
            )

    try:
        if training:
            from src.Axion.train import train

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
