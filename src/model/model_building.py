import pandas as pd
import numpy as np
import os
import yaml

import pickle

from sklearn.ensemble import RandomForestClassifier

n_estimators=yaml.safe_load(open("params.yaml","r"))["model_building"]["n_estimators"]

train_data = pd.read_csv("./data/processed/train_processed.csv")

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values



# X_train=train_data.drop(columns=['Potability'],axis=1)
# y_train=train_data['Potability']

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))