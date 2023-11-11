# Classification of Stars and Galaxies based on instrument readings 

# Data Pipeline Assignment



#### Prerequisites

Before using this script, make sure you have the following:

Airflow
DVC
GitHub
Visual Studio code
Python
Docker
Pytest

#### Data Preprocessing

Involves processing the given data to ensure the data is clean, free from outliers and Nan values,  avoiding the unnecessary features, making sure the data distribution is normal and splitting the data for training, testing and validation. Here few custom functions are written to preprocess the given data. We have created logs including the time, date of when the process has been initiated and when it got concluded along with the description of what function is called.

#### Testing

In testing, few functions has been written in pytest to ensure that our customs functions are valid and returns the true values before proceeding to the workflow, making sure the validity of data is consistent.

#### Workflows

Create DAGs using Apache airflow to schedule the preprocessing at specific time using a local host. Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

#### Data Versioning:

To keep of the track of the datasets that we work on, we use DVC. When we run the dvc cmd below it will store the datasets in the google cloud. The main advantage of using this is, we retrieve the older versions of the data. 


#### Data download:
https://github.com/jrspatel/Stellar_Proj/blob/main/dag/dags/data/star_classification.csv


#### Usage

-	Create a python virtual environment
-	Clone the github repository
-	DVC commands: dvc init, dvc repro, dvc dag
-   setup airflow using docker, import airflow in your virtual environment
-   Navigate to your local host ####
-   Once the DAG completes its execution, check any output or artifacts produced by functions and tasks.
-   Execute pytest commands for verification





```



# Data Pipeline Assignment



#### Prerequisites

Before using this script, make sure you have the following:

Airflow
DVC
GitHub
Visual Studio code
Python
Docker
Pytest

#### Data Preprocessing

Involves processing the given data to ensure the data is clean, free from outliers and Nan values,  avoiding the unnecessary features, making sure the data distribution is normal and splitting the data for training, testing and validation. Here few custom functions are written to preprocess the given data. We have created logs including the time, date of when the process has been initiated and when it got concluded along with the description of what function is called.

#### Testing

In testing, few functions has been written in pytest to ensure that our customs functions are valid and returns the true values before proceeding to the workflow, making sure the validity of data is consistent.

#### Workflows

Create DAGs using Apache airflow to schedule the preprocessing at specific time using a local host. Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

#### Data Versioning:

To keep of the track of the datasets that we work on, we use DVC. When we run the dvc cmd below it will store the datasets in the google cloud. The main advantage of using this is, we retrieve the older versions of the data. 





#### Usage

-	Create a python virtual environment
-	Clone the github repository
-	DVC commands: dvc init, dvc repro, dvc dag
-   setup airflow using docker, import airflow in your virtual environment
-   Navigate to your local host ####
-   Once the DAG completes its execution, check any output or artifacts produced by functions and tasks.
-   Execute pytest commands for verification





```
