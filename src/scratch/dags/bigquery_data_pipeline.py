# Python built-in modules
import os
import csv
from datetime import datetime, timedelta

# airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryGetDataOperator

# Slack
from slack_sdk import WebClient

data_dir = os.path.join(os.environ.get("AIRFLOW_HOME"), "./airflow-data")
os.makedirs(data_dir, exist_ok=True)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 7, 3),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# 데이터 저장 함수
def save_data_from_bigquery(**context):
    current_time_utc = datetime.utcnow()
    delta = timedelta(days=7)
    start_time = current_time_utc - delta
    start_time_unix = start_time.timestamp()

    # weekly data 추출
    review = task_get_review.execute(".")
    review = [input for input in review if float(input[1][:-6]) > start_time_unix]
    user_album = task_get_user_album.execute(".")
    user_album = [
        input for input in user_album if float(input[1][:-6]) > start_time_unix
    ]

    # weekly data xcom에 push
    task_instance = context["task_instance"]
    task_instance.xcom_push(key="review", value=review)
    task_instance.xcom_push(key="user_album", value=user_album)

    # 파일 이름 설정
    filename_review = (
        f"{data_dir}/{start_time.date()}~{current_time_utc.date()}-review.csv"
    )
    filename_user_album = (
        f"{data_dir}/{start_time.date()}~{current_time_utc.date()}-user_album.csv"
    )

    # csv로 저장하기
    with open(filename_review, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(review)

    with open(filename_user_album, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(user_album)


# weekly report 함수
def make_and_send_report(**context):
    slack_token = os.environ.get("SLACK_TOKEN")
    channel = "#weekly-report"
    weekly_review = context["task_instance"].xcom_pull(key="review")

    # 총 이용자수
    num_of_requests = len(weekly_review)

    # 평점 별 인원
    num_of_rating = [0, 0, 0, 0, 0, 0]
    total_sum = 0
    for review in weekly_review:
        if int(review[2]) >= 0:
            num_of_rating[int(review[2])] += 1
            total_sum += int(review[2])

    # 평균 평점
    mean_of_rating = total_sum / num_of_requests

    # 보고 날짜
    current_time_utc = datetime.utcnow()
    delta = timedelta(days=7)
    start_time = current_time_utc - delta

    # slack으로 보고
    client = WebClient(token=slack_token)
    client.chat_postMessage(
        channel=channel,
        text=f"Weekly Report \n --------------- \n 기간 : {start_time.date()} ~ {current_time_utc.date()} \n request 횟수 : {num_of_requests} \n 점수별 개수 : {num_of_rating}, \n 점수 평점 : {mean_of_rating}",
    )


with DAG(
    dag_id="bigquery_data_pipeline",
    default_args=default_args,
    schedule="@weekly",  # 일주일에 한번 실행
    tags=["bigquery"],
) as dag:
    # review
    task_get_review = BigQueryGetDataOperator(
        task_id="get_review",
        dataset_id="online_serving_logs",
        table_id="review",
        project_id="album-cover-generation-392406",
        max_results=100,
    )

    # user_album
    task_get_user_album = BigQueryGetDataOperator(
        task_id="get_user_album",
        dataset_id="online_serving_logs",
        table_id="user_album",
        project_id="album-cover-generation-392406",
        max_results=100,
    )

    task_save_data_from_bigquery = PythonOperator(
        task_id="save_data_from_bigqeury",
        python_callable=save_data_from_bigquery,
    )

    task_make_and_send_report = PythonOperator(
        task_id="make_and_send_report", python_callable=make_and_send_report
    )

    next_dag = TriggerDagRunOperator(
        task_id="trigger_model_retrain",
        trigger_dag_id="re_training",
    )

    (
        task_get_review
        >> task_get_user_album
        >> task_save_data_from_bigquery
        >> task_make_and_send_report
        >> next_dag
    )
