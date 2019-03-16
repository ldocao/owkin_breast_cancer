#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os
import pickle

import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split

from camelyon16 import TrainingPatients, TestPatients
from tile_predictor import LogisticRegressionL2


#load  data
test = TestPatients()
training = TrainingPatients()
features = training.stack_annotated_tile_features()
ground_truths = training.annotations["Target"].values

#cross validation for check
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('sampling', SMOTE()),
                     ('classification', LogisticRegressionL2().estimator)])

cv = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=0)
auc = cross_val_score(pipeline,
                      X=features, y=ground_truths,
                      cv=cv, scoring="roc_auc", verbose=0, n_jobs=-1)
auc = np.array(auc)
print(auc, auc.mean(), auc.std())



#apply standard scaler anyway
scaler = StandardScaler()
features = scaler.fit_transform(features)

#oversample positive labels
smt = SMOTE()
features, ground_truths = smt.fit_sample(features, ground_truths)
print(Counter(ground_truths))
clf = LogisticRegressionL2()
clf.train(features, ground_truths)



#predict over all tiles
tile_probas = clf.predict_tiles_of(training, scaler)
with open('predict_tile_training.pkl', 'wb') as handle:
    pickle.dump(tile_probas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#predict upon test tiles
test_probas = clf.predict_tiles_of(test, scaler)
with open('predict_tile_test.pkl', 'wb') as handle:
    pickle.dump(test_probas, handle, protocol=pickle.HIGHEST_PROTOCOL)

