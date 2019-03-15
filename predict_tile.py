#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from camelyon16 import TrainingPatients
from tile_predictor import LogisticRegressionL2, XGBoost


#training
features = TrainingPatients().stack_annotated_tile_features()
ground_truths = TrainingPatients().annotations["Target"].values


#cross validation
train_index, test_index = train_test_split(range(len(features)),
                                           train_size=0.8,
                                           stratify=ground_truths)
train_features = features[train_index, :]
train_gt = ground_truths[train_index]

validation_features = features[test_index, :]
validation_gt = ground_truths[test_index]

xgb = XGBoost().train(train_features, train_gt)
validation_proba = xgb.predict_proba(validation_features)
roc_auc_score(validation_gt, validation_proba[:, 1])
print(roc_auc_score)


