import numpy as np
import pandas as pd
import pickle
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load test data
test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# load model (FIX PATH)
model = pickle.load(open("models/model.pkl", "rb"))

# predictions
y_pred = model.predict(X_test)

# metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

metrics_dict = {
    "accuracy": float(acc),
    "precision": float(pre),
    "recall": float(recall),
    "f1_score": float(f1score)
}

os.makedirs("reports", exist_ok=True)

# save in correct location for DVC
with open("reports/metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)

print("Metrics saved successfully")