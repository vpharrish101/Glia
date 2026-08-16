import os

from minio import Minio

from src.utils.logging_mod import get_logger


logger=get_logger("DataLake")


class DataLake:
    def __init__(self):
        self.client=Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False,
        )

        self.bucket=os.getenv("MINIO_BUCKET")

        logger.info(
            "Data lake connected | bucket=%s",
            self.bucket,
        )

        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

            logger.info(
                "Data lake bucket created | bucket=%s",
                self.bucket,
            )
        else:
            logger.debug(
                "Data lake bucket available | bucket=%s",
                self.bucket,
            )

    def put(self,file_path,object_key):
        logger.info(
            "Uploading object | key=%s",
            object_key,
        )

        self.client.fput_object(
            self.bucket,
            object_key,
            str(file_path),
        )

        logger.info(
            "Object uploaded | key=%s",
            object_key,
        )

        return object_key