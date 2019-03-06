import numpy as np

from camelyon16 import TrainingPatients, TestPatients


training_patients = TrainingPatients()

features = []
for i in training_patients.ids():
    raw_resnet = training_patients.resnet_features(i)
    mean_resnet = np.mean(raw_resnet, axis=0)
    features.append(mean_resnet)
features = np.stack(features, axis=0)

ground_truths = training_patients.ground_truths()
