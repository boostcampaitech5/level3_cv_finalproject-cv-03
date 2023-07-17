# Google
from google.oauth2 import service_account
from google.cloud import bigquery


class BigQueryLogger:
    def __init__(self, gcp_config: dict):
        self.bigquery_config = gcp_config["bigquery"]

        self.project_id = gcp_config["project_id"]
        self.dataset_id = self.bigquery_config["dataset_id"]

        credentials = service_account.Credentials.from_service_account_file(
            gcp_config["credentials_path"]
        )

        self.client = bigquery.Client(credentials=credentials)

    def log(self, content: dict, id_name: str) -> None:
        table_id = f"{self.project_id}.{self.dataset_id}.{self.bigquery_config['table_id'][id_name]}"

        errors = self.client.insert_rows_json(table_id, [content])

        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
