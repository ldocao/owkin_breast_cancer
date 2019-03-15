import ipdb
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile



N_RESNET = 2048


annotated_tiles = TrainingPatients().annotations
annotated_patients = annotated_tiles["ID"].unique()
gt_tile = annotated_tiles["Target"].values


training_ids = TrainingPatients().ids
gt_patient = TrainingPatients().ground_truths["Target"].values 
n_patients = len(training_ids)

def train_tile():
    tiles = []
    for p in sorted(annotated_patients):
        resnet = TrainingPatients().resnet_features(p)
        tiles.append(resnet)
        features_tile = np.concatenate(tiles, axis=0)
        
    PARAMS = {"penalty": "l2",
              "C": 1.,
              "solver": "liblinear"}
    estimator = LogisticRegression(**PARAMS)
    estimator.fit(features_tile, gt_tile)
    return estimator


def predict_not_annotated_tile(estimator):
    results = {}
    for p in training_ids:
        resnet = TrainingPatients().resnet_features(p)
        n_tiles = resnet.shape[0]
        proba = estimator.predict_proba(resnet)[:,1]
        results[p] = proba
    return results


def construct_features_from_tile(features):
    QUARTILES = [25, 50, 75]
    x1 = np.mean(features)
    x2 = np.std(features)
    x3 = np.percentile(features, QUARTILES[0])
    x4 = np.percentile(features, QUARTILES[1])
    x5 = np.percentile(features, QUARTILES[2])
    x6 = np.max(features)
    x7 = np.min(features)
    x8 = (features > 0).sum() / len(features)
    return (x1, x2, x3, x4, x5, x6, x7, x8)
    

tile_estimator = train_tile()
predicted_tile_probas = predict_not_annotated_tile(tile_estimator)


features_train_patients = np.zeros([n_patients, 8])
for i in range(n_patients):
    p = training_ids[i]
    features_train_patients[i,:] = construct_features_from_tile(predicted_tile_probas[p])
    

aucs = []
N_RUNS = 3
for seed in range(N_RUNS):
    print(seed)
    PARAMS = {"penalty": "l2",
              "C": 1.,
              "solver": "liblinear"}
    estimator = LogisticRegression(**PARAMS)
    cv = StratifiedKFold(n_splits=5,
                         shuffle=True,
                         random_state=seed)

    auc = cross_val_score(estimator, X=features_train_patients, y=gt_patient,
                          cv=cv, scoring="roc_auc", verbose=0)
    aucs.append(auc)
aucs = np.array(aucs)
