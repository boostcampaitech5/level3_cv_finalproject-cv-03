# Python built-in modules
import os
import csv
from datetime import datetime, timedelta

# airflow
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryGetDataOperator

# Other modules
import pendulum


data_dir = os.path.join(os.environ.get("AIRFLOW_HOME"), "airflow-data")
os.makedirs(data_dir, exist_ok=True)

default_args = {
    "owner": "woojin",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2023, 7, 9, tz="UTC"),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="airflow-bigquery",
    default_args=default_args,
    schedule="0 0 * * 0",  # 일주일에 한번 실행
    tags=["bigquery"],
) as dag:
    # user_input
    get_user_input = BigQueryGetDataOperator(
        task_id="get_user_input",
        dataset_id="online_serving_logs",
        table_id="user_input",
        project_id="album-cover-generation-392406",
        max_results=100,
    )

    # user_review
    get_user_review = BigQueryGetDataOperator(
        task_id="get_user_review",
        dataset_id="online_serving_logs",
        table_id="user_review",
        project_id="album-cover-generation-392406",
        max_results=100,
    )

    get_user_input >> get_user_review

    current_time_utc = datetime.utcnow()
    delta = timedelta(days=7)
    start_time = current_time_utc - delta
    start_time_unix = start_time.timestamp()

    # 일주일 전 데이터만 저장
    user_input = get_user_input.execute(".")
    user_input = [
        input for input in user_input if float(input[0][:-6]) > start_time_unix
    ]
    user_reivew = get_user_review.execute(".")
    user_review = [
        input for input in user_reivew if float(input[1][:-6]) > start_time_unix
    ]

    # 파일 이름 설정
    filename_user_input = (
        f"{data_dir}/{start_time.date()}~{current_time_utc.date()}-user_input.csv"
    )
    filename_user_review = (
        f"{data_dir}/{start_time.date()}~{current_time_utc.date()}-user_review.csv"
    )

    # csv로 저장하기
    with open(filename_user_input, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(user_input)

    with open(filename_user_review, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(user_reivew)
