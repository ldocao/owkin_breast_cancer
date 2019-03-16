import ipdb

import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from camelyon16 import TrainingPatients
from patient_predictor import LogisticRegressionL2
from tile_feature import TileFeature
from challenge import Challenge


#load training
with open('predict_tile_training.pkl', 'rb') as handle:
    training_predictions = pickle.load(handle)

training_ids = sorted(list(training_predictions.keys()))
n_patients = len(training_ids)
gt_patients = TrainingPatients().ground_truths["Target"].values 


#construct feature matrix
N_FEATURES = 9 #deduced from TileFeature.engineer
features_patients = np.zeros([n_patients, N_FEATURES])
for i in range(n_patients):
    p = training_ids[i]
    tile_probas = training_predictions[p] #tile prediction values
    features_patients[i,:] = TileFeature(tile_probas).engineer()

scaler = StandardScaler()
features_patients = scaler.fit_transform(features_patients)

# smt = SMOTE()
# features_patients, gt_patients = smt.fit_sample(features_patients, gt_patients)
lr = LogisticRegressionL2()
lr.train(features_patients, gt_patients)



# predict on real test set
with open('predict_tile_test.pkl', 'rb') as handle:
    test_predictions = pickle.load(handle)

test_ids = sorted(list(test_predictions.keys()))
n_patients = len(test_ids)

features_patients = np.zeros([n_patients, N_FEATURES])
for i in range(n_patients):
    p = test_ids[i]
    tile_probas = test_predictions[p]
    features_patients[i,:] = TileFeature(tile_probas).engineer()

features_patients = scaler.transform(features_patients)    
predictions = lr.predict(features_patients)
Challenge(test_ids, predictions).submit("predict_patient5.csv")

