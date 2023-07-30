# Python built-in modules
import io

# Google
from google.cloud import storage
from google.oauth2 import service_account

# datetime
from datetime import datetime, timedelta


class GCSUploader:
    def __init__(self, gcp_config: dict):
        credentials_path = gcp_config["credentials_path"]
        self.bucket_name = gcp_config["cloud_storage"]["bucket_name"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = storage.Client(credentials=credentials)

    def upload_blob(self, source_file_data: bytes, destination_blob_name: str) -> str:
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        file_obj = io.BytesIO(source_file_data)
        blob.upload_from_file(file_obj)

        # permanent_url = f"https://storage.cloud.google.com/{self.bucket_name}/{destination_blob_name}"

        user_expiration_time = datetime.now() + timedelta(days=30)
        user_url = blob.generate_signed_url(expiration=user_expiration_time)

        print(f"File uploaded to {destination_blob_name}.")

        return user_url

    # Uploads image to GCS and returns the URL
    def save_image_to_gcs(self, urls: list) -> str:
        image_urls = []
        for i, (byte_arr, url_name) in enumerate(urls):
            url = self.upload_blob(byte_arr, url_name)
            image_urls.append(url)

        return image_urls
