import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

from challenge import Challenge
from camelyon16 import TrainingPatients, TestPatients


def mean_resnet(dataset):
    features = []
    for i in dataset.ids():
        raw_resnet = dataset.resnet_features(i)
        mean_resnet = np.mean(raw_resnet, axis=0)
        features.append(mean_resnet)
    features = np.stack(features, axis=0)
    return features


# training set
print("loading training set")
training_patients = TrainingPatients()
features_train = mean_resnet(training_patients)
ground_truths = training_patients.ground_truths

#test set
print("loading test set")
test_patients = TestPatients()
features_test = mean_resnet(test_patients)


# Train a final model on the full training set
print("train logistic regression")
PARAMS = {"penalty": "l2",
          "C": 1.,
          "solver": "liblinear"}
estimator = sklearn.linear_model.LogisticRegression(**PARAMS)
estimator.fit(features_train, ground_truths[Challenge.TARGET_COLUMN].values)
preds_test = estimator.predict_proba(features_test)[:, 1]

# dump submission
Challenge(test_patients.ids(), preds_test).submit("reproduce_baseline.csv")
