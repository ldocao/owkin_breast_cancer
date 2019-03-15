import ipdb
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile


training_ids = TrainingPatients().ids
n_training = len(training_ids)
Y_train = TrainingPatients().ground_truths["Target"].values 


def features(i):
    print(i)
    r = TrainingPatients().resnet_features(i)
    x1 = np.mean(r, axis=0)
    x2 = np.std(r, axis=0)
    x3 = np.min(r, axis=0)
    x4 = np.max(r, axis=0)
    x = np.array([x1, x2, x3, x4])
    return x


X_train = np.zeros([n_training, 4, 2048])
for i in range(n_training):
    X_train[i] = features(training_ids[i])
    
X_train = np.reshape(X_train, [n_training, 4*2048])


# grid search for hyper parameters
# print("grid search")
# grid = {'n_estimators' : np.linspace(500, 2000, 5, dtype=int),
#         'max_depth' : np.linspace(5, 100, 5, dtype=int),
#         "max_features": np.linspace(2, 200, 5, dtype=int)}
# rfc = RandomForestClassifier(n_jobs=-1)
# grid_search = GridSearchCV(rfc, grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, Y_train)
# grid_search.best_params_
best_params = {'max_depth': 100, 'max_features': 150, 'n_estimators': 500}


aucs = []
N_RUNS = 3
for seed in range(N_RUNS):
    print(seed)
    estimator = RandomForestClassifier(**best_params, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=seed)

    auc = cross_val_score(estimator, X=X_train, y=Y_train,
                          cv=cv, scoring="roc_auc", verbose=0)
    aucs.append(auc)
aucs = np.array(aucs)

print("Predicting weak labels by mean resnet")
print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))
