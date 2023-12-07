import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from mlflow.models.signature import infer_signature

# Load your dataset
# For this example, let's assume you have a DataFrame named 'data' with features and target variable 'target'
# X = data.drop('target', axis=1)
# y = data['target']
def load_data_modelling(cols):
    data_train = pd.read_csv('mlflow/data-train-X_train.csv')
    data_test = pd.read_csv('mlflow/data-test-X_test.csv')
    remove_cols = cols
    data_train = data_train.drop(columns = remove_cols, axis=1)
    y_train = data_train['class']
    y_train = y_train.replace({'GALAXY':0, 'QSO':1, 'STAR':2})
    X_train = data_train.drop(columns=['class'], axis =1)
    data_test = data_test.drop(columns = remove_cols, axis=1)
    y_test = data_test['class']
    y_test = y_test.replace({'GALAXY':0, 'QSO':1, 'STAR':2})
    X_test = data_test.drop(columns=['class'], axis =1)    
    return X_train, X_test, y_train, y_test 
# Split the dataset into training and testing sets
remove_cols = ['obj_ID', 'run_ID', 'rerun_ID', 'field_ID', 'spec_obj_ID', 'fiber_ID']
X_train, X_test, y_train, y_test = load_data_modelling(remove_cols)

# Define the logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=10)

# Define hyperparameter grid for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a new model with the best hyperparameters
best_model = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=10, **best_params)

# Train the model on the full training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)


# Log the metrics and parameters with MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("R2-Score", accuracy)
    # mlflow.log_metric("precision", precision)
    # mlflow.log_metric("f1", f1)
    #mlflow.log_artifact("your_dataset.csv")  # Log your dataset for reproducibility
    #mlflow.sklearn.log_model(best_model, "model")
    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path='sklearn' + 'Logistic Regression',
            signature=signature,
            registered_model_name="Log-Reg-Model",
        )

# Print the results
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("R Report:\n", classification_rep)
# print("F1-Score:\n", classification_rep)

