from datetime import datetime
from google.oauth2 import service_account
from google.cloud import bigquery
import yaml


class BigQueryLogger:
    def __init__(self):
        with open("config.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)

        bigquery_config = config["bigquery"]
        credentials_path = bigquery_config["credentials_path"]
        project_id = bigquery_config["project_id"]
        dataset_id = bigquery_config["dataset_id"]
        table_id = bigquery_config["table_id"]
        self.table_id_full = f"{project_id}.{dataset_id}.{table_id}"

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = bigquery.Client(credentials=credentials)

    def log(self, album, summarization, request_id, image_urls):
        row_to_insert = [
            {
                "request_id": request_id,
                "request_time": datetime.utcnow().isoformat(),
                "song_names": album.song_names,
                "artist_name": album.artist_name,
                "genre": album.genre,
                "album_name": album.album_name,
                "release": album.release,
                "lyric": album.lyric,
                "summarization": summarization,
                "image_urls2": image_urls,
            }
        ]
        errors = self.client.insert_rows_json(self.table_id_full, row_to_insert)
        if errors:
            print(f"Encountered errors while inserting rows: {errors}")


# Instantiate BigQueryLogger outside of the class definition
# Use it by just importing bigquery_logger in main
# def main():
#     bigquery_logger = BigQueryLogger()

# if __name__ == '__main__':
#     main()
