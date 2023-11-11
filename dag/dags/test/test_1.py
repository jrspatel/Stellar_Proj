import os
import pandas as pd
import pytest
from datapreprocessing import data_stats, split_dataset, scaling, checking_NaN
import random
import numpy as np

#checking the type of the columns
def test_data_stats():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    columns = df.columns
    print(columns)
    print(1)
    assert len(columns) == 18

def test_split_dataset():
    x_train, x_test, y_train, y_test = split_dataset()
    assert x_train.shape == (80000, 18) 
    assert y_test.shape == (20000, )

def test_data_stat():
    df = data_stats()
    assert len(df.columns) <= 18

def test_check_nan():
    df = checking_NaN()
    a = len(df.isna().any())
    assert a > 0

def test_data_stats_1():
    df = data_stats()
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    columns = df.columns
    assert len(columns) != 0


def test_data_stats_3():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    columns = df.columns
    assert len(columns) >= 0

def test_scaling():
    # Create a temporary test CSV file

    test_data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'class': [0, 1, 0]
    }
    test_df = pd.DataFrame(test_data)
    test_csv_path = 'test_data.csv'
    test_df.to_csv(test_csv_path, index=False)

    # Call the scaling function on the test data
    scaling(test_csv_path)

    # Read the scaled data from the CSV file
    scaled_data = pd.read_csv(test_csv_path)

    # Assertions to check if the scaling is applied correctly
    assert 'class' in scaled_data.columns, "Class column is missing in the scaled data"
    assert scaled_data.shape == (3, 3), "Scaled data has incorrect shape"

