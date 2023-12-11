import os
import pandas as pd
import pytest
from datapreprocessing import data_stats, scaling, checking_NaN
import random
import numpy as np
import gcsfs



# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem(project='stellarclassification', token='C:\Users\Vdhya\Downloads\stellarclassification-3587e05b548a.json')

def test_split_dataset():
    X_train_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(X_train_path, 'r') as f:
        X_train = pd.read_csv(f)
    X_test_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    with fs.open(X_test_path, 'r') as f:
        X_test = pd.read_csv(f)
    assert x_train.shape == (80000, 18) 
    assert X_test.shape == (20000, 18)

def test_data_stat():
    df = data_stats()
    assert len(df.columns) <= 18

# def test_check_nan():
#     df = checking_NaN()
#     a = len(df.isna().any())
#     assert a > 0

# def test_data_stats_1():
#     df = data_stats()
#     columns = df.columns
#     assert len(columns) != 0


# def test_data_stats_3():
#     data_hash = "bce06389286124270180cc96bf116584"  # Replace with the actual hash
#     df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
#     columns = df.columns
#     assert len(columns) >= 0

def test_scaling():
    # Create a temporary test CSV file

    test_data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'class': [0, 1, 0]
    }
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'pytest_entries.csv'
    test_df.to_csv(test_csv_path, index=False)

    # Call the scaling function on the test data
    scaling(test_csv_path)

    # Read the scaled data from the CSV file
    scaled_data = pd.read_csv(test_csv_path)

    # Assertions to check if the scaling is applied correctly
    assert 'class' in scaled_data.columns, "Class column is missing in the scaled data"
    assert scaled_data.shape == (3, 3), "Scaled data has incorrect shape"