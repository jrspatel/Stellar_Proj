import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score
import joblib, scipy
from imblearn.over_sampling import SMOTE
import json
from collections import Counter
import gcsfs

fs = gcsfs.GCSFileSystem()

def drop_column(dataset):
 
    dataset.drop(['g','z','rerun_ID','spec_obj_ID','i','obj_ID'],axis=1,inplace=True)
    dataset['MJY'] = dataset.MJD / 365.0
    dataset.drop('MJD',axis=1,inplace=True)
    dataset_temp_path = "gs://stellarclassification_bucket/data/temp/temp_dataset.csv"
    if not dataset.empty:
        with fs.open(dataset_temp_path, 'w') as f:
            dataset.to_csv(f, index=False)
    print("drop dataset is completed")
    return dataset
    
def outlier_detection(dataset):
    for col in dataset.drop('class',axis=1).columns:
        lower_limit, upper_limit = dataset[col].quantile([0.25,0.75])
        IQR = upper_limit - lower_limit
        lower_whisker = lower_limit - 1.5 * IQR
        upper_whisker = upper_limit + 1.5 * IQR
        dataset[col] = np.where(dataset[col]>upper_whisker,upper_whisker,np.where(dataset[col]<lower_whisker,lower_whisker,dataset[col]))
    print("Outlier dataset is completed")
    return dataset

def oversampling_class(dataset):
    X = dataset.drop('class',axis=1)
    y = dataset['class']
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X,y)
    print("Oversampling dataset is completed")
    return X_smote, y_smote

def scaling_dataset(X, y):
    scaler = StandardScaler()
    features = X.columns
    X_scale = scaler.fit_transform(X)
    X_scale = pd.DataFrame(X_scale,columns=features)
    mean_values = X.mean()
    std_deviation = X.std()
    normalization_stats = {
        'mean': mean_values.to_dict(),
        'std': std_deviation.to_dict()
    }
    scaling_stats_gcs_path = "gs://stellarclassification_bucket/data_stats/stats.json"

    with fs.open(scaling_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)
    
    return X_scale, y

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test

def main():
    dataset_gcs_path = "gs://stellarclassification_bucket/data/star_classification.csv"
    X_train_gcs_path = "gs://stellarclassification_bucket/data/train/X_train.csv"
    y_train_gcs_path = "gs://stellarclassification_bucket/data/train/y_train.csv"
    X_test_gcs_path = "gs://stellarclassification_bucket/data/test/X_test.csv"
    y_test_gcs_path = "gs://stellarclassification_bucket/data/test/y_test.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)
    droped_column_ds = drop_column(dataset)
    out_ds = outlier_detection(droped_column_ds)
    X, y = oversampling_class(out_ds)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train, y_train = scaling_dataset(X_train,y_train)
    if not X_train.empty:
        with fs.open(X_train_gcs_path, 'w') as f:
            X_train.to_csv(f, index=False)
    if not X_test.empty:
        with fs.open(X_test_gcs_path, 'w') as f:
            X_test.to_csv(f, index=False)
    if not y_train.empty:
        with fs.open(y_train_gcs_path, 'w') as f:
            y_train.to_csv(f, index=False)
    if not y_test.empty:
        with fs.open(y_test_gcs_path, 'w') as f:
            y_test.to_csv(f, index=False)
main()