import mlflow.pyfunc
import pandas as pd

# Load the model


model = mlflow.pyfunc.load_model("/mlruns/0/82addde92aeb4ec38e78c661f6fb05b9/artifacts/rf-model/model.pkl")

# Make predictions
predictions = model.predict([1.2376623035210803e+18,246.974161866776,33.1523252429912,21.96427,20.25309,19.0962,18.56822,18.19525,3918,301,6,324,1.2287190463224764e+19,0.2813719,10913,58256,890,'GALAXY'])
