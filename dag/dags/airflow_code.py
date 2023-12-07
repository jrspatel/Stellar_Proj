from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import sys
import os
# Add the path to the directory containing the source
sys.path.append('/path/to/directory/containing/source')

from src.datapreprocessing import split_dataset, checking_NaN, data_stats, scaling, outlier_elimination
from src.random_forest_model import rf_hyperoptimization 
default_args = {
    'owner': 'Team 3',
    'retries': 5,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    default_args=default_args,
    dag_id="preprocessing_v1",
    description="Data Preprocessing",
    start_date=datetime(2023, 11, 8),
    schedule_interval='@daily',
    catchup=True,
) as dag:

    split_task = PythonOperator(
        task_id = 'split_task',
        python_callable = split_dataset,
        provide_context = True,
        # In split_task

        dag = dag
    )

    check_nan_task = PythonOperator(
    task_id='check_nan_task',
    python_callable=checking_NaN,
    provide_context=True,
    dag = dag,
)
    
    stats_task = PythonOperator(
        task_id = 'stats_task',
        python_callable = data_stats,
        provide_context=True,
        # In split_task
        dag = dag
    )

    scaling_task = PythonOperator(
        task_id = 'scaling_task',
        python_callable = scaling,
        provide_context=True,
        
        # In split_task

        dag = dag
    )

    outlier_detection_task = PythonOperator(
        task_id = 'outlier_detection_task',
        python_callable = outlier_elimination,
        provide_context=True,
        
        # In split_task

        dag = dag
    )
    RF_optimization_task = PythonOperator(
        task_id = 'RF_optimization_task',
        python_callable = rf_hyperoptimization,
        provide_context=True,
        
        # In split_task

        dag = dag
    )

    #import_data_task >> split_task

    split_task >> check_nan_task >> stats_task >> scaling_task >> outlier_detection_task >> RF_optimization_task
    # split_task >> stats_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()

