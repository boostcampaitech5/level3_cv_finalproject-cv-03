# Python built-in modules
import io

# Google
from google.cloud import storage
from google.oauth2 import service_account

# datetime
from datetime import datetime, timedelta


class GCSUploader:
    def __init__(self, gcp_config):
        credentials_path = gcp_config["credentials_path"]
        self.bucket_name = gcp_config["cloud_storage"]["bucket_name"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = storage.Client(credentials=credentials)

    # Uploads image to GCS and returns the URL
    def save_image_to_gcs(self, byte_arr, destination_blob_name):
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)

        file_obj = io.BytesIO(byte_arr)
        blob.upload_from_file(file_obj)

        # permanent_url = f"https://storage.cloud.google.com/{self.bucket_name}/{destination_blob_name}"

        user_expiration_time = datetime.now() + timedelta(days=30)
        user_url = blob.generate_signed_url(expiration=user_expiration_time)

        print(f"File uploaded to {destination_blob_name}.")

        return user_url
