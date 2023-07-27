# Python built-in modules
import os
import pandas as pd
from datetime import datetime, timedelta

# airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# 재학습에 필요한 파일 만들기
def make_data_for_retraining():
    # 현재 날짜 계산
    current_time_utc = datetime.utcnow()
    delta = timedelta(days=7)
    start_time = current_time_utc - delta

    # 경로 설정
    save_dir = f"/opt/ml/input/code/level3_cv_finalproject-cv-03/stable_diffusion/experiments/[retrain]{current_time_utc.date()}"
    read_dir = f"/opt/ml/input/code/level3_cv_finalproject-cv-03/src/scratch/airflow-data/{start_time.date()}~{current_time_utc.date()}-review.csv"

    # 경로 만들기
    os.makedirs(save_dir, exist_ok=True)

    # weekly review 가져오기
    df = pd.read_csv(read_dir)
    df = df[df.iloc[:, 1] >= 3]
    text = (
        df.iloc[:, 5] + " " + df.iloc[:, 6] + " " + df.iloc[:, 7] + " " + df.iloc[:, 8]
    )
    album = df.iloc[:, 4]
    df = pd.concat([album, text], axis=1)
    df.columns = ["img_url", "text"]

    # weekly review 저장
    df.to_csv(save_dir + "/albums.csv", index=False)

    # prompts.txt
    with open(save_dir + "/prompts.txt", mode="w") as file:
        file.write("album of dance\n")
        file.write("album of hiphop")


default_args = {
    "owner": "woojin",
    "depends_on_past": False,
    "start_date": datetime(2023, 7, 3),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="re_training",
    default_args=default_args,
    tags=["bigquery"],
) as dag:
    task_make_data_for_retraining = PythonOperator(
        task_id="make_data_for_retraining",
        python_callable=make_data_for_retraining,
    )

    # 학습 파일 실행
    current_time_utc = datetime.utcnow().date()
    task_re_train = BashOperator(
        task_id="task_re_train",
        bash_command=f"python -m stable_diffusion.main --exp-name=[retrain]{current_time_utc} --max-train-steps=1 --valid-epoch=1 --ckpt-step=1 --grid-size 1 2",
        cwd="/opt/ml/input/code/level3_cv_finalproject-cv-03/src",
        dag=dag,
    )

    task_make_data_for_retraining >> task_re_train
