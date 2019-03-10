import ipdb

import numpy as np

import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input


from camelyon16 import TrainingPatients, AnnotatedTile



BATCH_SIZE = 32

#resnet
INPUT_SHAPE = (224, 224, 3)
resnet = ResNet50(include_top=False,
                  weights='imagenet',
                  input_tensor=None,
                  input_shape=INPUT_SHAPE,
                  pooling="max")

#raw data
training_patients = TrainingPatients()
annotations = training_patients.annotations
filenames = annotations.index.values
target = annotations["Target"].values
train_files, test_files = train_test_split(filenames, test_size=0.2, stratify=target)




def preprocess(filename):
    """Returns numpy array preprocessed for ResNet50

    Parameters
    ----------
    filename: str
        name of file to load (not absolute path)
    """
    path = AnnotatedTile(filename).path
    img = image.load_img(path, target_size=INPUT_SHAPE[:2])
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x


def chunker(seq, size):
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))


preds = []
ids = []
i = 0
ratio = len(train_files) // BATCH_SIZE
for batch in chunker(train_files, BATCH_SIZE):
    print(f"batch {i} / {ratio}")
    X = [preprocess(str(s)) for s in batch]
    X = np.array(X)
    preds_batch = resnet.predict(X)
    preds += preds_batch.tolist()
    ids += list(batch)
    i += 1

preds = np.array(preds) #features for each tile


# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=1024, whiten=True)
pca = pca.fit(preds)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(preds)


PARAMS = {"penalty": "l2",
          "C": 1.,
          "solver": "liblinear"}
estimator = sklearn.linear_model.LogisticRegression(**PARAMS)
estimator.fit(X_train, annotations.loc[train_files]["Target"].values)


preds = []
ids = []
i = 0
ratio = len(test_files) // BATCH_SIZE
for batch in chunker(test_files, BATCH_SIZE):
    print(f"batch {i} / {ratio}")
    X = [preprocess(str(s)) for s in batch]
    X = np.array(X)
    preds_batch = resnet.predict(X)
    preds += preds_batch.tolist()
    ids += list(batch)
    i += 1

preds = np.array(preds) #features for each tile
X_test = pca.transform(preds)

test_predicts = estimator.predict(X_test)
y_test = annotations.loc[test_files]["Target"]

auc = roc_auc_score(y_test, test_predicts)
print("auc score", auc)
