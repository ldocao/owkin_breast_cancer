#PURPOSE: use retrained resnet, predict over tiles
import os
import cv2
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile

DIR = "/home/ldocao/owkin/tensorboard/run_002/"
BEST_MODEL = os.path.join(DIR, "weights-improvement-17-0.95.hdf5")
model = load_model(BEST_MODEL)
INPUT_SHAPE = (96, 96)
BATCH_SIZE = 32
training_tiles = TrainingPatients().tiles
test_tiles = TestPatients().tiles
tiles = pd.concat([training_tiles, test_tiles])



def chunker(seq, size):
    return (seq[pos: pos+size] for pos in range(0, len(seq), size))


preds = {}
n_batch = len(tiles) // BATCH_SIZE
count = 0
for batch in chunker(tiles.index, BATCH_SIZE):
    print(count, n_batch)
    X = [preprocess_input(cv2.resize(cv2.imread(str(AnnotatedTile(x).path)), INPUT_SHAPE)) for x in batch]
    X = np.array(X)
    preds_batch = model.predict(X)
    d = {k: p[0] for k, p in zip(list(batch), preds_batch)}
    preds = {**preds, **d}
    count += 1


predictions = pd.DataFrame.from_dict(preds, orient="index", columns=["probability"])
predictions.to_csv(os.path.join(DIR, "prediction_tiles.csv"))
