"""
Created By: Team 03
Date: November 2, 2023
"""
#importing necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Set up logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)


log_filename = f"log_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = TimedRotatingFileHandler(log_filename, when='midnight', backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

logger = logging.getLogger("logger")
logger.addHandler(file_handler)

def data_stats():
    logger.info("Calculating data statistics...")
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    
    variable_stats = dataset.describe()
    logger.info("Data statistics calculated.")
    return variable_stats

def split_dataset():
    logger.info("Splitting dataset...")
    data_hash = "bce06389286124270180cc96bf116584"  # Replace with the actual hash

# Use the hash in the data path

    data_path = f"data/my_dataset.csv.{data_hash}.dvc"
    dataset = pd.read_csv(data_path)
    #dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    y = dataset['class']
    X = dataset.drop(columns = ['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
    X_train['class'] = y_train.to_list()
    X_test['class'] = y_test.to_list()
    
    X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    
    X_test.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_test.csv"), index=False)
    path = "dags/data/star_classification.csv"
    logger.info("Dataset split.")
    return 1

def checking_NaN():
    logger.info("Checking for NaN values...")
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    nan_rows = len(dataset[dataset.isna().any(axis=1)])
    
    if nan_rows != 0:
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        dataset = dataset.drop(labels = nan_index, axis=0)
        dataset = dataset.reset_index(drop = True)
    
    dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train_nan.csv"), index=False)
    logger.info("NaN values checked and handled.")

    return 1

def scaling():
    logger.info("Scaling dataset...")
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train_nan.csv"))
    scaling = MinMaxScaler()
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    X_train_columns = X_train.columns
    X_train_array = scaling.fit_transform(X = X_train)
    X_train = pd.DataFrame(X_train_array)
    X_train.columns = X_train_columns
    X_train['class'] = y
    
    X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train_scaling.csv"), index=False)
    logger.info("Dataset scaled.")
    return 1

def outlier_elimination():

    logger.info("Eliminating outliers...")
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train_scaling.csv"))
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    threshold = 1.0

    outliers = ((X_train < (Q1 - threshold * IQR)) | (X_train > (Q3 + threshold * IQR))).any(axis=1)

    dataset_no_outliers = X_train[~outliers]
    dataset_no_outliers['class'] = y
    
    logger.info("Outliers eliminated.")

    os.system(f"dvc add {dataset_no_outliers}")
    os.system("dvc commit -f")
    return 1

def pcadatset(X_t):
    """
        @Parameters:
            X_t(DataFrame): Train dataset
        @does:
            Perform PCa
        @Returns:
            X_train(Dataset): Dataset after removing few columns
    """
    logger.info("Performing PCA on the dataset...")
    principal = PCA(n_components=3)
    principal.fit(X_t)
    x = principal.transform(X_t)
    logger.info("PCA performed.")
    return x