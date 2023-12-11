import os
import pandas as pd
import pytest
from datapreprocessing import data_stats, scaling, checking_NaN
import random
import numpy as np


def test_data_stat():
    df = data_stats()
    assert len(df.columns) <= 3

def test_check_nan():
    df = checking_NaN()
    a = len(df.isna().any())
    assert a > 0

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
    scaled_data = scaling()
    # Assertions to check if the scaling is applied correctly
    assert 'class' in scaled_data.columns, "Class column is missing in the scaled data"
    assert scaled_data.shape == (3, 3), "Scaled data has incorrect shape"