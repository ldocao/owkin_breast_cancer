#PURPOSE: predict using trained nasnet mobile
# LINK: https://www.kaggleusercontent.com/kf/7550002/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K6yzwa3j_XisokCxORn-XA.htbY50oVPJZMWtcb3g7LrzrvVsAgKiH67PjqqkYuUaHrTKJ4ycTy68sSbq2sT8gqMyiWWjcXjmU3qo4KPE0hxNHiCMt-MBWoG58zjw8LhOdv7eHF4nQ7m0SsJAmezMgq_kbaMfEDnqBSlW2W8CeJtlENCL-kvixkx3qM9zHz1kY.Uu_iAvTotyFHAckg7t_KmA/model.h5

import os
import cv2

import numpy as np
import pandas as pd

from keras.models import load_model
from keras.applications.nasnet import preprocess_input
from sklearn.metrics import roc_auc_score

from kaggle import TrainingImages, TestImages
from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile

USE_CPU = False
BATCH_SIZE = 32

if USE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""



#model
print("load model")
TRAINED_NASNET = "/home/ldocao/owkin/kaggle/models/trained_nasnet_mobile.h5"
INPUT_SHAPE = (96, 96)
nasnet = load_model(TRAINED_NASNET, compile=False) #use compile option https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi



def chunker(seq, size):
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))


#whole owkin training data
training = TrainingPatients().tiles
training["dataset"] = "training"

preds = {}
n_batch = len(training) // BATCH_SIZE
count = 0
for batch in chunker(training.index, BATCH_SIZE):
    print(count, n_batch)
    X = [preprocess_input(cv2.resize(cv2.imread(x), INPUT_SHAPE)) for x in batch]
    X = np.array(X)
    preds_batch = nasnet.predict(X)
    d = {k: p[0] for k, p in zip(list(batch), preds_batch)}
    preds = {**preds, **d}
    count += 1

preds = pd.DataFrame.from_dict(preds, orient="index")
training["prediction_probas"] = preds[0]
predictions = training.groupby("patient_id")[["prediction_probas"]].max()
gt = TrainingPatients().ground_truths
print(roc_auc_score(gt["Target"].values, predictions))

#tile level training data
annotated_tiles = TrainingPatients().annotations

preds = {}
n_batch = len(annotated_tiles) // BATCH_SIZE
count = 0
for batch in chunker(annotated_tiles.index, BATCH_SIZE):
    print(count, n_batch)
    X = [preprocess_input(cv2.resize(cv2.imread(str(AnnotatedTile(x).path)), INPUT_SHAPE)) for x in batch]
    X = np.array(X)
    preds_batch = nasnet.predict(X)
    d = {k: p[0] for k, p in zip(list(batch), preds_batch)}
    preds = {**preds, **d}
    count += 1

preds = pd.DataFrame.from_dict(preds, orient="index")
annotated_tiles["prediction_probas"] = preds[0]
print(roc_auc_score(annotated_tiles["Target"], annotated_tiles["prediction_probas"]))
