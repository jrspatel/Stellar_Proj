#importing necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()
  
def data_stats():
    #dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"))
    dataset_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)

    variable_stats = dataset.describe()
    print("Data stats is completed")
    return variable_stats

def split_dataset():
    dataset_gcs_path = "gs://stellarclassification_bucket/data/star_classification.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)
    y = dataset['class']
    X = dataset.drop(columns = ['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
    X_train['class'] = y_train.to_list()
    X_test['class'] = y_test.to_list()

    X_train_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    y_train_gcs_path = "gs://stellarclassification_bucket/data/train/y_train.csv"
    X_test_gcs_path = "gs://stellarclassification_bucket/data/test/X_test.csv"
    y_test_gcs_path = "gs://stellarclassification_bucket/data/test/y_test.csv"

    # Save the updated datasets to GCS
    if not X_train.empty:
        with fs.open(X_train_gcs_path, 'w') as f:
            X_train.to_csv(f, index=False)
    if not y_train.empty:
        with fs.open(y_train_gcs_path, 'w') as f:
            y_train.to_csv(f, index=False)
    if not X_test.empty:
        with fs.open(X_test_gcs_path, 'w') as f:
            X_test.to_csv(f, index=False)
    if not y_test.empty:
        with fs.open(y_test_gcs_path, 'w') as f:
            y_test.to_csv(f, index=False) 
    print("Split dataset is completed") 
    return 1

def checking_NaN():
    dataset_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)
    nan_rows = len(dataset[dataset.isna().any(axis=1)])
   
    if nan_rows != 0:
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        dataset = dataset.drop(labels = nan_index, axis=0)
        dataset = dataset.reset_index(drop = True)

    if not dataset.empty:
        with fs.open(dataset_gcs_path, 'w') as f:
            dataset.to_csv(f, index=False)
    #dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
        print("Check nan is completed")
    return 

def scaling():
    dataset_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)

    scaling = MinMaxScaler()
    y = dataset['class']
    X_train = dataset.drop(['class'], axis=1)
    X_train_columns = X_train.columns
    X_train_array = scaling.fit_transform(X = X_train)
    X_train = pd.DataFrame(X_train_array)
    X_train.columns = X_train_columns
    X_train['class'] = y
    
    if not X_train.empty:
        with fs.open(dataset_gcs_path, 'w') as f:
            X_train.to_csv(f, index=False)
        print("Scaling is completed")
    #X_train.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)

    return 1

def outlier_elimination():
    # Assuming dataset is a pandas DataFrame
    dataset_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)

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
    #dataset_no_outliers['class'] = y

    if not dataset_no_outliers.empty:
        with fs.open(dataset_gcs_path, 'w') as f:
            dataset_no_outliers.to_csv(f, index=False)
        print("Outlier is completed")    
    #dataset_no_outliers.to_csv(os.path.join(os.path.dirname(__file__), "../data/X_train.csv"), index=False)
    return 1

# def main():
#     data_stats()
#     split_dataset()
#     checking_NaN()
#     scaling()

# if __name__ == '__main__':
#     main()
