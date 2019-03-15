import os
import pickle
import numpy as np
import pandas as pd

from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile
from patient_predictor import LogisticRegressionL2


training_ids = TrainingPatients().ids
gt_patients = TrainingPatients().ground_truths["Target"].values 
n_patients = len(training_ids)




with open('predict_tile.pkl', 'rb') as handle:
    tile_predictions = pickle.load(handle)


def construct_features_from_tile(features):
    QUARTILES = [25, 50, 75]
    x1 = np.mean(features)
    x2 = np.std(features)
    x3 = np.percentile(features, QUARTILES[0])
    x4 = np.percentile(features, QUARTILES[1])
    x5 = np.percentile(features, QUARTILES[2])
    x6 = np.max(features)
    x7 = np.min(features)
    x8 = (features > 0.5).sum() / len(features)
    return (x1, x2, x3, x4, x5, x6, x7, x8)



features_train_patients = np.zeros([n_patients, 8])
for i in range(n_patients):
    p = training_ids[i]
    features_train_patients[i,:] = construct_features_from_tile(tile_predictions[p])
    

N_RUN = 4
aucs = []
for i in range(N_RUN):    
    lr = LogisticRegressionL2()
    auc = lr.cross_validation(features_train_patients, gt_patients, i)
    aucs.append(auc)


