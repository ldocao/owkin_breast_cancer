# PURPOSE: predict metastases presence using resnet features with SVM

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
Y_train = ground_truths[Challenge.TARGET_COLUMN].values

#test set
print("loading test set")
test_patients = TestPatients()
features_test = mean_resnet(test_patients)



# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(features_train)
X_train = ss.transform(features_train)
X_test = ss.transform(features_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=128, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# Train classifier and obtain predictions for OC-SVM
# print("grid search")
# grid = {'gamma' : np.logspace(-9, 3, 50),
#         'C' : np.logspace(-4, 1, 50),
#         "kernel": ("linear", "rbf")}

# grid_search = GridSearchCV(SVC(), grid, cv=5, n_jobs=-1)
# grid_search.fit(X_train, Y_train)
# grid_search.best_params_


# classifier using grid search results
best_params = {'C': 0.013894954943731374,
               'gamma': 1.0000000000000001e-09,
               'kernel': 'linear'}
clf = SVC(**best_params, probability=True)
clf.fit(X_train, Y_train)
results = clf.predict_proba(X_test)
results = [x[0] for x in results]
Challenge(test_patients.ids(), results).submit("predict_svm.csv")


