#importing necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

  
def data_stats():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../test/pytest_entries.csv"))
    variable_stats = dataset.describe()
    print("Data stats is completed")
    return variable_stats

def split_dataset():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../test/pytest_entries.csv"))
    y = dataset['class']
    X = dataset.drop(columns = ['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
    X_train['class'] = y_train.to_list()
    X_test['class'] = y_test.to_list()
    print("Split dataset is completed") 
    return X_train, X_test, y_train, y_test

def checking_NaN():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../test/pytest_entries.csv"))
    nan_rows = len(dataset[dataset.isna().any(axis=1)])
    if nan_rows != 0:
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        dataset = dataset.drop(labels = nan_index, axis=0)
        dataset = dataset.reset_index(drop = True)
    return dataset

def scaling():
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../test/pytest_entries.csv"))
    scaling = MinMaxScaler()
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    X_train_columns = X_train.columns
    X_train_array = scaling.fit_transform(X = X_train)
    X_train = pd.DataFrame(X_train_array)
    X_train.columns = X_train_columns
    X_train['class'] = y
    return X_train
