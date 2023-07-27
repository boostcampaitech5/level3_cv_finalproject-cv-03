# Python built-in modules
import io

# Google
from google.cloud import storage
from google.oauth2 import service_account


class GCSUploader:
    def __init__(self, gcp_config: dict):
        credentials_path = gcp_config["credentials_path"]
        self.default_bucket_name = gcp_config["cloud_storage"]["bucket_name"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = storage.Client(credentials=credentials)

    def upload_blob(
        self,
        source_file_data: bytes,
        destination_blob_name: str,
        bucket_name: str = None,
    ) -> str:
        bucket_name = bucket_name if bucket_name else self.default_bucket_name
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        file_obj = io.BytesIO(source_file_data)
        blob.upload_from_file(file_obj)

        print(f"File uploaded to {destination_blob_name} in {bucket_name}.")
        return blob.public_url

    # Uploads image to GCS and returns the URL
    def save_image_to_gcs(self, urls: list, bucket_name: str = None) -> str:
        image_urls = []
        for i, (byte_arr, url_name) in enumerate(urls):
            url = self.upload_blob(byte_arr, url_name, bucket_name)
            image_urls.append(url)

        return image_urls
