# Google
from google.oauth2 import service_account
from google.cloud import error_reporting


class ErrorReporter:
    def __init__(self, gcp_config):
        credentials_path = gcp_config["credentials_path"]

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )

        self.client = error_reporting.Client(credentials=credentials)

    def python_error(self):
        self.client.report_exception()
