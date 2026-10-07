import logging
import os
import sys

from src.ETL.preprocessing import run_etl
from src.train import train
from src.inference import run_inference

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",)

def main():
    setup_logging()
    logger=logging.getLogger("Glia.Orchestrator")
    config_path="config.yaml"
    logger.info(f"Using config: {config_path}")

    try:
        run_etl(config_path)
        train(config_path)
        run_inference(config_path)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__=="__main__":
    main()
