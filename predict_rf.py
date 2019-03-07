# PURPOSE: predict metastases presence using resnet features with RF classifier

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


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


# my data set
print("loading training set")
training_patients = TrainingPatients()
features_train = mean_resnet(training_patients)
ground_truths = training_patients.ground_truths
Y_train = ground_truths[Challenge.TARGET_COLUMN].values




#real test set
print("loading test set")
test_patients = TestPatients()
features_test = mean_resnet(test_patients)




# grid search for hyper parameters
print("grid search")
grid = {'n_estimators' : np.linspace(500, 2000, 5, dtype=int),
        'max_depth' : np.linspace(5, 100, 5, dtype=int),
        "max_features": np.linspace(2, 200, 5, dtype=int)}
rfc = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(rfc, grid, cv=5, n_jobs=-1)
grid_search.fit(features_train, Y_train)
grid_search.best_params_


# cross val score
# print("cross val score")
# N_RUNS = 5
# aucs = []
# for seed in range(N_RUNS):
#     estimator = RandomForestClassifier(**best_params, probability=True)
#     cv = StratifiedKFold(n_splits=5,
#                          shuffle=True,
#                          random_state=seed)
#     auc = cross_val_score(estimator,
#                           X=features_train,
#                           y=Y_train,
#                           cv=cv, scoring="roc_auc", verbose=0)
#     aucs.append(auc)

# aucs = np.array(aucs)
# print(auc.mean(), auc.std())


# # classifier using grid search results
# clf = SVC(**best_params, probability=True)
# clf.fit(features_train, Y_train)
# results = clf.predict_proba(X_test)
# results = [x[0] for x in results]
# Challenge(test_patients.ids(), results).submit("predict_rf.csv")


