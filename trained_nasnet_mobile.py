#PURPOSE: predict using trained nasnet mobile
# LINK: https://www.kaggleusercontent.com/kf/7550002/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..K6yzwa3j_XisokCxORn-XA.htbY50oVPJZMWtcb3g7LrzrvVsAgKiH67PjqqkYuUaHrTKJ4ycTy68sSbq2sT8gqMyiWWjcXjmU3qo4KPE0hxNHiCMt-MBWoG58zjw8LhOdv7eHF4nQ7m0SsJAmezMgq_kbaMfEDnqBSlW2W8CeJtlENCL-kvixkx3qM9zHz1kY.Uu_iAvTotyFHAckg7t_KmA/model.h5

import os
import cv2

import numpy as np

from keras.models import load_model
from keras.applications.nasnet import preprocess_input
from sklearn.metrics import roc_auc_score

from kaggle import TrainingImages, TestImages


USE_CPU = False
BATCH_SIZE = 32

if USE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""



#model
print("load model")
TRAINED_NASNET = "/home/ldocao/owkin/kaggle/models/trained_nasnet_mobile.h5"
nasnet = load_model(TRAINED_NASNET, compile=False) #use compile option https://stackoverflow.com/questions/53740577/does-any-one-got-attributeerror-str-object-has-no-attribute-decode-whi


#kaggle data
print("load data")
training = TrainingImages().training
validation = TrainingImages().validation
gt = TrainingImages().ground_truths

def chunker(seq, size):
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))

preds = []
ids = []
PATH = "/home/ldocao/owkin/kaggle/train/"

for batch in chunker(training, BATCH_SIZE):
    X = [preprocess_input(cv2.imread(PATH+x+".tif")) for x in batch]
    X = np.array(X)
    preds_batch = nasnet.predict(X)
    preds += preds_batch.T.tolist()[0]
    ids += list(batch)
    

print(roc_auc_score(gt.loc[training]["label"].values, preds)) #check AUC on training set, must be close to 1
#Out[4]: 0.9927753500420746



#owkin data
