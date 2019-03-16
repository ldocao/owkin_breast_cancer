#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os
import pickle

import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from camelyon16 import TrainingPatients, TestPatients
from tile_predictor import LogisticRegressionL2


#load training data
training = TrainingPatients()
features = training.stack_annotated_tile_features()
ground_truths = training.annotations["Target"].values

#apply standard scaler anyway
scaler = StandardScaler()
features = scaler.fit_transform(features)


#create validation set
x_train, x_test, y_train, y_test = train_test_split(features, ground_truths,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=ground_truths, random_state=1)

clf = LogisticRegressionL2()
clf.train(x_train, y_train)
print(Counter(y_train))
p_test = clf.predict(x_test)
print("without smote AUC", roc_auc_score(y_test, p_test))


#oversample positive labels
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)
print(Counter(y_train))
clf = LogisticRegressionL2()
clf.train(x_train, y_train)
p_test = clf.predict(x_test)
print("with smote AUC", roc_auc_score(y_test, p_test))

ipdb.set_trace()

#train classifier
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

