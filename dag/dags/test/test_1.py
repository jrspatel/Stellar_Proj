from io import StringIO
from unittest.mock import patch, MagicMock, call
from datapreprocessing import drop_column, outlier_detection, oversampling_class, scaling_dataset, split_dataset
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def generate_mock_data(num_rows=15):
    np.random.seed(42)  # For reproducibility
    
    # Generate mock data with specified columns
    data = {
        'obj_ID': np.arange(1, num_rows + 1),
        'alpha': np.random.uniform(0, 1, num_rows),
        'delta': np.random.uniform(0, 1, num_rows),
        'u': np.random.uniform(20, 25, num_rows),
        'g': np.random.uniform(19, 22, num_rows),
        'r': np.random.uniform(18, 21, num_rows),
        'i': np.random.uniform(17, 19, num_rows),
        'z': np.random.uniform(16, 18, num_rows),
        'run_ID': np.arange(101, 101 + num_rows),
        'rerun_ID': np.arange(201, 201 + num_rows),
        'cam_col': np.random.choice([1, 2], num_rows),
        'field_ID': np.arange(11, 11 + num_rows),
        'spec_obj_ID': np.arange(301, 301 + num_rows),
        'class': np.random.choice([0, 1], num_rows),
        'redshift': np.random.uniform(0, 0.5, num_rows),
        'plate': np.arange(401, 401 + num_rows),
        'MJD': np.arange(501, 501 + num_rows),
        'fiber_ID': np.arange(601, 601 + num_rows)
    }
    mock_dataset = pd.DataFrame(data)
    return mock_dataset

def test_outlier_detection():
    mock_data = generate_mock_data()
    result = outlier_detection(mock_data)
    assert len(result) == len(mock_data)
    assert list(result.columns) == list(mock_data.columns)


def local_oversampling_class(dataset, k_neighbors=1):
    X = dataset.drop('class', axis=1)
    y = dataset['class']    
    # Use a fixed number of neighbors for testing purposes
    smote = SMOTE(k_neighbors=k_neighbors)   
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote

def test_oversampling_class():
    mock_data = generate_mock_data()
    result_X, result_y = local_oversampling_class(mock_data, k_neighbors=1)
    assert len(result_X) > len(mock_data)
    assert len(result_y) > len(mock_data)


def test_scaling_dataset():
    mock_data = generate_mock_data()
    mock_X = mock_data.drop('class', axis=1)
    mock_y = mock_data['class']
    result_X, result_y = scaling_dataset(mock_X, mock_y)
    assert len(result_X) == len(mock_data)
    assert len(result_y) == len(mock_data)
    assert set(result_X.columns) == set(mock_X.columns)


def test_split_dataset():
    mock_data = generate_mock_data()
    mock_X = mock_data.drop('class', axis=1)
    mock_y = mock_data['class']    
    result_X_train, result_X_test, result_y_train, result_y_test = split_dataset(mock_X, mock_y)    
    assert len(result_X_train) > 0
    assert len(result_X_test) > 0
    assert len(result_y_train) > 0
    assert len(result_y_test) > 0


# Run tests
if __name__ == '__main__':
    test_outlier_detection()
    test_oversampling_class()
    test_scaling_dataset()
    test_split_dataset()