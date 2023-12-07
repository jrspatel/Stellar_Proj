#importing necessary Libraries
import pandas as pd
import os
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()


def dataset_download_gcs():
    #Download dataset from GCS
    dataset_gcs_path = "gs://stellarclassification_bucket/data/star_classification.csv"
    with fs.open(dataset_gcs_path, 'r') as f:
        dataset = pd.read_csv(f)
        
    #Save downloaded dataset in local      
    dataset.to_csv(os.path.join(os.path.dirname(__file__), "../data/star_classification.csv"))
    return 1


def main():
    dataset_download_gcs()

if __name__ == '__main__':
    main()