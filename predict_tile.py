#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from camelyon16 import TrainingPatients, TestPatients
from tile_predictor import LogisticRegressionL2


#load training data
training = TrainingPatients()
features = training.stack_annotated_tile_features()
ground_truths = training.annotations["Target"].values


#train classifier
scaler = StandardScaler()
features = scaler.fit_transform(features)
clf = LogisticRegressionL2()
clf.train(features, ground_truths)


#predict over all tiles
tile_probas = clf.predict_tiles_of(training, scaler)

with open('predict_tile_training.pkl', 'wb') as handle:
    pickle.dump(tile_probas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#predict upon test tiles
test = TestPatients()
test_probas = clf.predict_tiles_of(test, scaler)

with open('predict_tile_test.pkl', 'wb') as handle:
    pickle.dump(test_probas, handle, protocol=pickle.HIGHEST_PROTOCOL)

