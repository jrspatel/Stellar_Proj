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

  
def data_stats():
    dataset = pd.resoud_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    variable_stats = dataset.describe()
    return variable_stats

def split_dataset():

    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    y = dataset['class']
    X = dataset.drop(columns = ['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
    X_train['class'] = y_train.to_list()
    X_test['class'] = y_test.to_list()
    
    X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    
    X_test.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_test.csv"), index=False)
    path = "dags/data/star_classification.csv"
    return 1

def checking_NaN():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    nan_rows = len(dataset[dataset.isna().any(axis=1)])
    
    if nan_rows != 0:
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        dataset = dataset.drop(labels = nan_index, axis=0)
        dataset = dataset.reset_index(drop = True)
    
    dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return 1

def scaling():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    scaling = MinMaxScaler()
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    X_train_columns = X_train.columns
    X_train_array = scaling.fit_transform(X = X_train)
    X_train = pd.DataFrame(X_train_array)
    X_train.columns = X_train_columns
    X_train['class'] = y
    
    X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return 1

def outlier_elimination():
    # Assuming dataset is a pandas DataFrame
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    # Define a threshold to identify outliers
    threshold = 1.0

    # Identify outliers
    outliers = ((X_train < (Q1 - threshold * IQR)) | (X_train > (Q3 + threshold * IQR))).any(axis=1)

    # Remove outliers
    dataset_no_outliers = X_train[~outliers]
    dataset_no_outliers['class'] = y
    
    dataset_no_outliers.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
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
    principal = PCA(n_components=3)
    principal.fit(X_t)
    x = principal.transform(X_t)
    return x