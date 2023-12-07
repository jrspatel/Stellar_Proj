<p align="center">
    <b>
        <h1 align="center"> Classification Of Celestial Objects Using Sloan Digital Sky Survey Data</h1>
    </b>
</p>

Following are the main contents to follow:

>   -  [Introduction](#project-intro)<br>
>   -  [Dataset Information](#dataset-details)<br>
>       -  [Download Dataset](#download-dataset)<br>
>   -  [Initial Set-Up](#setup-)<br>
>   -  [Data Preprocessing](#data-preprocessing)<br>
>   -  [Models (Performance Metrics)](#models-)<br>
>   -  [Hyperparameter Tuning](#hyperparameter-tuning)<br>
>   -  [Results](#results-)<br>
>   -  [Applications Used](#tools-)<br>
>   -  [Conclusion](#conclusion-)<br>


### Introduction<a id='project-intro'></a>

Machine learning has found extensive applications within the realm of space research and
exploration, causing a profound transformation in the methodologies employed by scientists for data interpretation, predictive modeling, and the enhancement of spacecraft operations.

In 2007, the astrophysicist Kevin Schawinski confronted a substantial challenge when presented with an enormous dataset comprising images of galaxies sourced from the Sloan Digital Sky Survey. The process of manually classifying each galaxy as either elliptical or spiral consumed a considerable amount of his time. Recognizing the impracticality of this approach, Schawinski, along with Chris Lintott, conceived the Galaxy Zoo citizen science project. This initiative engaged the participation of over 100,000 members of the general public, resulting in a substantial reduction in the time required to complete the task. Even with the invaluable assistance of citizen scientists, it still took two years to complete the categorization of all the images.

The introduction of cutting-edge devices, such as the Large Synoptic Survey Telescope and the Dark Energy Spectroscopic Instrument, has led to the generation of datasets that surpass human processing capabilities. Consequently, astrophysicists have increasingly embraced Artificial Intelligence as the optimal solution to address a multitude of critical challenges encountered in their research endeavors. 

With that being stated, the central objective of our project is to conceptualize, develop, and implement a machine-learning model tailored specifically for the categorization of celestial entities into three distinct classes: Galaxy, Star, and Quasar. This classification will be based on a comprehensive dataset acquired from the Sloan Digital Sky Survey. The dataset predominantly comprises various characteristics of celestial objects, such as color, shape, and spatial distribution, as observed by the SDSS telescope. Detailed information regarding this dataset will be expounded upon in the subsequent section.


#### Dataset Details<a id='dataset-details'></a>

The dataset, a structured representation from the Sloan Digital Sky Survey, is sourced from the publicly accessible SDSS RD17 release. It is a vital resource for advancing our understanding of the universe's structure, development, and fundamental governing principles, playing a pivotal role in contemporary astrophysics and cosmology research.

The dataset comprises 100,000 rows with 18 columns: 17 features and 1 class column. These variables provide crucial information about observed objects, including celestial coordinates, photometric properties, spectroscopic characteristics, and observational details. The important features that are taken into consideration are:
 
1. alpha: Right Ascension angle (at J2000 epoch), indicating the object's celestial coordinates.
2. delta: Declination angle (at J2000 epoch), specifying the object's celestial coordinates.
3. u: Ultraviolet filter in the photometric system, representing a specific spectral band.
4. g: Green filter in the photometric system, corresponding to a particular spectral band.
5. r: Red filter in the photometric system, indicative of a specific spectral band.
6. i: Near-infrared filter in the photometric system, associated with a particular spectral band.
7. z: Infrared filter in the photometric system, denoting a specific spectral band.
8. cam_col: Camera column, used to identify the scanline within the specified run.
9. redshift: Redshift value, determined based on the observed increase in wavelength.
10. plate: Plate ID, providing identification for each individual plate used in SDSS.
11. MJD: Modified Julian Date, serving as a timestamp to indicate when a particular piece of SDSS data was acquired.
12. class: Object class, which can be categorized as a galaxy, star, or quasar object.

Sample images of Galaxy, Star and Quasar
![Image](Stellar_Proj/assets/Galaxy_Star_Quasar.png)


#### Data download<a id='download-dataset'></a>

<pre>
Dataset              : <a href=https://storage.cloud.google.com/airflow-data-bucket-practice-group3/star_classification.csv>Stellar Classification Dataset</a>   
</pre>


#### Initial Set-Up<a id='setup'></a>

Set-up a python virtual environment
    ```
    python -m venv <environment_name>
    ```

Activate the virtual environment
    ```
    <environment_name>\Scripts\activate
    ```

Run the following command to clone the GitHub repository into the current directory
    ```
    git clone <repository_url>
    ```

Run the requirements.txt file to install all the dependencies
    ```
      pip install -r requirements.txt
    ```

Run the load_data.py file to download the dataset into the directory
    ```
      python load_data.py
    ```


#### Data Preprocessing<a id='data-preprocessing'></a>

Involves processing the given data to ensure the data is clean, free from outliers and Nan values,  avoiding the unnecessary features, making sure the data distribution is normal and splitting the data for training, testing and validation. Here few custom functions are written to pre-process the given data. We have created logs including the time, date of when the process has been initiated and when it got concluded along with the description of what function is called.

Here is a simple flow chart depicting the data pre-processing flow

```mermaid
graph TD;
    Split Data-->Remove NaN
    Remove NaN-->Store Statistics;
    Store Statistics-->Scale Data;
```

#### Models (Performance Metrics)<a id='models-'></a>

We applied three multi-class classification models to the pre-processed data, obtaining the following results. The models used are Random Forest Classifier, Decision Tree classifier, XGBoost and Logistic Regression.

<pre>
<b> Random Forest Classifier </b>
Test Accuracy                    : 97.8%
F1-score                         : 97.7%
R2-score                         : 96.0%

<b> Decision Tree classifier </b>
Test Accuracy                    : 97.0%
F1-score                         : 97.0%
R2-score                         : 94.7%

<b> XGBoost </b>
Test Accuracy                    : 97.5%
F1-score                         : 97.5%
R2-score                         : 94.7%

<b> Logistic Regression </b>
Test Accuracy                    : 94.0%
F1-score                         : 94.0%
R2-score                         : 94.7%
</pre>

Based on the above results, we move forward with Random Forest Classifier as it seems to be the best model.


#### Hyperparameter Tuning<a id='hyperparameter-tuning'></a>

For the tuning of the models we are using a technique called hyperopt. The fmin(fn=objective(), params = best_params ..) runs for the number of trails and give out the best params with the maximum accuracy. 

```bash
def objective():
    .......
    ....... 
    return {'loss': -accuracy , 'params':params }
```

We employ the subsequent parameters in the hyperparameter tuning process.
<pre>
n_estimators          : Number of Trees
max_depth             : Tree Depth
criterion             : Gini Index or Entropy
</pre>

```bash
params_rf = OrderedDict([
    ('n_estimators', hp.randint('n_estimators', 100, 200)),
    ('criterion', hp.choice('criterion', ['gini', 'entropy'])),
    ('max_depth', hp.randint('max_depth', 10, 30))])
```
![Image](Stellar_Proj/assets/n_estimators.png)![Image](Stellar_Proj/assets/max_depth.png)![Image](Stellar_Proj/assets/criterion.png)


#### Results<a id='results-'></a>

From the given ranges of hyperparameter, the following provide the optimal results
<pre>
n_estimators          : 107
max_depth             : 20
criterion             : Entropy
</pre>

<b>Performance</b>
<pre>
Test Accuracy                    : 97.5%
F1-score                         : 97.5%
R2-score                         : 94.7%
</pre>


#### Applications Used<a id='tools-'></a>

1. Python 
2. Data Version Control (DVC) 
3. Docker
4. Machine learning algorithms
5. MLFlow
6. Google Cloud Storage
7. Airflow
8. Visual Studio Code


#### Conclusion<a id='conclusion-'></a>

F1 score is critical for assessing the efficacy of a stellar classification model, particularly whilst class imbalance exists. It provides a balanced assessment of precision and recall, which is critical in astronomy because the classes of celestial objects may be represented in the dataset to varied degrees. The F1 score can help researchers make informed decisions about the model's efficacy and applicability for their unique astronomical studies.

Therefore, we adopt the F1 score as the metric to assess the model's performance. The Random Forest Classifier emerges as the optimal model, exhibiting the highest F1 score.


























```