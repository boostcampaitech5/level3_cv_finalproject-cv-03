from google.cloud import storage
from google.oauth2 import service_account

import os
import io
import yaml


class GCSUploader:
    def __init__(self):
        with open("config.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)

        gcs_config = config["gcs"]
        credentials_path = gcs_config["credentials_path"]
        self.bucket_name = gcs_config["bucket_name"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = storage.Client(credentials=credentials)

    def upload_blob(self, source_file_data, destination_blob_name):
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        file_obj = io.BytesIO(source_file_data)
        blob.upload_from_file(file_obj)

        print(f"File uploaded to {destination_blob_name}.")
        return blob.public_url

    # Uploads image to GCS and returns the URL
    def save_image_to_gcs(self, image, destination_blob_name):
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        image_bytes = byte_arr.getvalue()

        # Upload the image data to GCS
        url = self.upload_blob(image_bytes, destination_blob_name)

        return url
