#importing necessary Libraries
import pandas as pd
import os
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()


def dataset_upload_gcs():
    X_train = pd.read_csv('Stellar_Proj/mlflow/X_train.csv')
    y_train = pd.read_csv('Stellar_Proj/mlflow/y_train.csv')
    X_test = pd.read_csv('Stellar_Proj/mlflow/X_test.csv')
    y_test = pd.read_csv('Stellar_Proj/mlflow/y_test.csv')
    
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


def main():
    dataset_upload_gcs()

if __name__ == '__main__':
    main()