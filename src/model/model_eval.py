import numpy as np
import pandas as pd
import pickle
import json
import os
import mlflow
import dagshub


dagshub.init(repo_owner='sivakumar026', repo_name='water_quality_mlpipeline', mlflow=True)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load test data
test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# load model
model = pickle.load(open("models/model.pkl", "rb"))

# predictions
y_pred = model.predict(X_test)

# metrics
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# ---------------- IMPORTANT FIX ----------------
mlflow.set_tracking_uri("https://dagshub.com/sivakumar026/water_quality_mlpipeline.mlflow")
mlflow.set_experiment("water_quality_pipeline")

# ---------------- MLflow logging ----------------
with mlflow.start_run() as run:

    print("RUN ID:", run.info.run_id)

    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_metric("precision", float(pre))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("f1_score", float(f1score))

    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_param("n_estimators", model.n_estimators)

# ---------------- DVC metrics ----------------
metrics_dict = {
    "accuracy": float(acc),
    "precision": float(pre),
    "recall": float(recall),
    "f1_score": float(f1score)
}

os.makedirs("reports", exist_ok=True)

with open("reports/metrics.json", "w") as file:
    json.dump(metrics_dict, file, indent=4)

print("Metrics saved successfully")