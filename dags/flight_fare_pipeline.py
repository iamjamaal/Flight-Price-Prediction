"""
flight_fare_pipeline.py
Airflow DAG that orchestrates the end-to-end flight fare prediction
ML pipeline from data ingestion through model interpretation.

Schedule: Weekly (can also be triggered manually at any time).

Task dependency graph:
    load_and_validate
          │
    clean_and_preprocess
        ┌───┴───┐
   eda_report  baseline_model     ← run in parallel
        └───┬───┘
    train_advanced_models
          │
    interpret_and_report
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline import (
    load_and_validate,
    clean_and_preprocess,
    generate_eda_report,
    train_baseline_model,
    train_advanced_models,
    interpret_and_report,
)

default_args = {
    "owner": "flight-fare-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="flight_fare_prediction",
    default_args=default_args,
    description="End-to-end ML pipeline: ingest → clean → EDA → train → interpret",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "flight-fare", "regression"],
) as dag:

    load = PythonOperator(
        task_id="load_and_validate",
        python_callable=load_and_validate,
    )

    preprocess = PythonOperator(
        task_id="clean_and_preprocess",
        python_callable=clean_and_preprocess,
    )

    eda = PythonOperator(
        task_id="generate_eda_report",
        python_callable=generate_eda_report,
    )

    baseline = PythonOperator(
        task_id="train_baseline_model",
        python_callable=train_baseline_model,
    )

    advanced = PythonOperator(
        task_id="train_advanced_models",
        python_callable=train_advanced_models,
    )

    interpret = PythonOperator(
        task_id="interpret_and_report",
        python_callable=interpret_and_report,
    )

    # EDA and baseline training run in parallel after preprocessing
    load >> preprocess >> [eda, baseline] >> advanced >> interpret
