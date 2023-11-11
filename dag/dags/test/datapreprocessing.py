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
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    variable_stats = dataset.describe()
    return variable_stats

def split_dataset():
    data_path = "data/my_dataset.csv"
    data = pd.read_csv(data_path)
    y = dataset['class']
    X = dataset.drop(columns = ['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
    X_train['class'] = y_train.to_list()
    X_test['class'] = y_test.to_list()
    
    X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    
    X_test.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_test.csv"), index=False)
    path = "dags/data/star_classification.csv"
    return  X_train, X_test, y_train, y_test

def checking_NaN():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    nan_rows = len(dataset[dataset.isna().any(axis=1)])
    
    if nan_rows != 0:
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        dataset = dataset.drop(labels = nan_index, axis=0)
        dataset = dataset.reset_index(drop = True)
    
    dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return dataset

def scaling(dataset):
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    scaling = MinMaxScaler()
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    X_train_columns = X_train.columns
    for i in X_train_columns:
        X_train_array = scaling.fit_transform(X = X_train)
        X_train = pd.DataFrame(X_train_array)
    X_train['class'] = y
    
    dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return dataset

def outlier_elimination():
    # Assuming dataset is a pandas DataFrame
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)
    IQR = Q3 - Q1

    # Define a threshold to identify outliers
    threshold = 1.5

    # Identify outliers
    outliers = ((X_train < (Q1 - threshold * IQR)) | (X_train > (Q3 + threshold * IQR))).any(axis=1)

    # Remove outliers
    dataset_no_outliers = X_train[~outliers]
    dataset_no_outliers['class'] = y

    dataset_no_outliers.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return dataset

def test_outlier_elimination():

    test_data = {
        'feature1': [1, 2, 3, 100],  
        'feature2': [4, 5, 6, 7],
        'class': [0, 1, 0, 1]
    }
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'test_data.csv'
    test_df.to_csv(test_csv_path, index=False)


    outlier_elimination()

    result_data = pd.read_csv(test_csv_path)


    assert 'class' in result_data.columns, "Class column is missing in the result data"
    assert result_data.shape[0] == 3, "Outlier was not properly removed"
