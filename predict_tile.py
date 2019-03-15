#PURPOSE: predict at tile-level on TrainingPatients

import ipdb
import os
import pickle


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from camelyon16 import TrainingPatients
from tile_predictor import LogisticRegressionL2


#training
features = TrainingPatients().stack_annotated_tile_features()
ground_truths = TrainingPatients().annotations["Target"].values

#grid search
clf = LogisticRegressionL2()
# best_params = clf.grid_search(features, ground_truths)
# print(best_params)


#predict upon all tiles
clf.train(features, ground_truths) #train over all annotated tiles
patient_ids = TrainingPatients().tiles["patient_id"].unique()
patient_ids = [str(i).zfill(3) for i in patient_ids]
tile_probas = clf.predict_tiles_of(patient_ids)


with open('predict_tile.pkl', 'wb') as handle:
    pickle.dump(tile_probas, handle, protocol=pickle.HIGHEST_PROTOCOL)
