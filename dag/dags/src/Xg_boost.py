import pandas as pd
import mlflow
import scipy.optimize as opt
from functools import partial
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, r2_score, f1_score
from mlflow.tracking.client import MlflowClient
from xgboost import XGBClassifier
import mlflow
from mlflow.models.signature import infer_signature
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp, space_eval   
from collections import OrderedDict
import os
import csv
import json
import scipy.optimize as opt
from functools import partial
# from loggers import logger

def load_data_modelling():
    cols = ['obj_ID', 'run_ID', 'rerun_ID', 'field_ID', 'spec_obj_ID', 'fiber_ID']
    data_train = pd.read_csv('Stellar_Proj/mlflow/data-train-X_train.csv')
    data_test = pd.read_csv('Stellar_Proj/mlflow/data-test-X_test.csv')
    remove_cols = cols
    data_train = data_train.drop(columns=remove_cols, axis=1)
    y_train = data_train['class']
    y_train = y_train.replace({'GALAXY': 0, 'QSO': 1, 'STAR': 2})
    X_train = data_train.drop(columns=['class'], axis=1)
    data_test = data_test.drop(columns=remove_cols, axis=1)
    y_test = data_test['class']
    y_test = y_test.replace({'GALAXY': 0, 'QSO': 1, 'STAR': 2})
    X_test = data_test.drop(columns=['class'], axis=1)
    return X_train, X_test, y_train, y_test

def objective(params, X, y):
    print('=== model_name XGBOOST ====')
    print("====params=======", params)
   

    # Create and train models.
    X_train, X_test , y_train, y_test = load_data_modelling()
    print("============== shapes ===============")
    print(X_train.shape, y_train.shape)

    model = XGBClassifier(**params)
    print('ok')
    
    # logger.info('>>>>>>>>>>>>>> Model Loaded sucessfully ...')
    print('model running', model)
    model.fit(X_train, y_train)
    # Use the model to make predictions on the test dataset.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    f1_score_var = f1_score(y_test, predictions, average='weighted')
    print(predictions)
    # logger.info('>>>>>>>>>>>>>> Model prediction don
    # e sucessfully ...')
    # Log the sklearn model and register as version 1
        
    
    #return {'loss': -accuracy_score(y_test, predictions) ,'params':params,'status': STATUS_OK}
    return {'loss': -accuracy, 'accuracy': accuracy, 'r2': r2, 'f1_score': f1_score_var, 'params': params, 'status': STATUS_OK}



# Your objective function
def model_registry(params, X, y):
    # ... (model training and evaluation)
    X_train, X_test, y_train, y_test = load_data_modelling()
    print("============== Registring the model in MLFLOW ===============")
    print(X_train.shape, y_train.shape)
    print(params)
    #djncofkd
    model = XGBClassifier(**params)
    # n_estimators = 100, min_sample_split = 20
    model.fit(X_train,y_train)
    # Calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    f1_score_var = f1_score(y_test, predictions, average='weighted')
    another_experiment_name = 'AnotherExperiment'
    mlflow.set_experiment(another_experiment_name)
    # Log parameters
    with mlflow.start_run() as run:
        mlflow.log_params(params)  # Log the parameters
        mlflow.log_metric("accuracy", accuracy)  # Log the accuracy
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("f1_score", f1_score_var)
        signature = infer_signature(X_test, predictions)
        # Register the model with MLflow
        mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xg-model",  # Specify your desired artifact path
                registered_model_name="best_xg",
                signature=signature,  # Include if you have a signature
            )
    
    # return {'loss': -accuracy, 'status': STATUS_OK}
def dt_hyperoptimization():
    X_train, X_test, y_train, y_test = load_data_modelling()
    print(X_train.shape, y_train.shape)

    params_xg = {'n_estimators': hp.randint('n_estimators',100, 200),
                'min_sample_split' : hp.randint('min_sample_split',100,200)
                }


    print("====================== Executing {} ==================".format(' MODEL'))
    space = params_xg
        
    partial_function = partial(objective, X= X_train, y=y_train)
    
    result = fmin(fn=partial_function, space=params_xg, algo=tpe.suggest,max_evals=10)  # x0 is the initial guess 
    best_params = space_eval(space, result)

    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_sample_split'] = int(best_params['min_sample_split'])
    params = {'XGBoost': best_params}
    with open(os.path.join("Stellar_Proj/mlflow", "model_param_xg.json"), 'r') as json_file:
        data = json.load(json_file)
    best_params = (data['XGBoost'])
    
    model_registry(best_params, X_train, y_train)
    client = MlflowClient()
    client.update_registered_model(
        name="best_xg",
        description="This XGBoost model data is used to classify star, galaxy. The data consists of 18 features"
    )
dt_hyperoptimization()
