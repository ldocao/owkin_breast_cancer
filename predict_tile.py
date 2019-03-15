#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os

import numpy as np
import pandas as pd

from camelyon16 import TrainingPatients
from tile_predictor import LogisticRegressionL2


#training
features = TrainingPatients().stack_annotated_tile_features()
ground_truths = TrainingPatients().annotations["Target"].values
lr = LogisticRegressionL2()
estimator = lr.train(features, ground_truths)


#cross validation
auc = lr.cross_validation(features, ground_truths)
print("auc", auc)
print("auc mean/std", auc.mean(), auc.std())

