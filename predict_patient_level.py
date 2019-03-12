#PURPOSE: use predictions at tile-level and use DBSCAN to predict at patient level
# the idea is to use probas as a 3rd dimension, and clusterize. we discard every cluster with less than say 4 members (arbitrary?) to discard eventual wrong positive labeling. Then we take average over members of each cluster, and take the max

import ipdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score
from camelyon16 import TrainingPatients, TestPatients, AnnotatedTile

DIR = "/home/ldocao/owkin/tensorboard/run_002"
PREDICTION_TILES = os.path.join(DIR, "prediction_tiles.csv")
tile_probas = pd.read_csv(PREDICTION_TILES, names=["path", "p"], skiprows=1)
tile_probas.set_index("path", inplace=True)
training_tile_infos = TrainingPatients().tiles
test_tile_infos = TestPatients().tiles
tile_infos = pd.concat([training_tile_infos, test_tile_infos])
tiles = tile_infos.merge(tile_probas, how="left", right_index=True, left_index=True)
tiles = tiles[["patient_id", "x", "y", "p"]]


#rescale x,y to [0,1], just like p
tiles["x_norm"] = minmax_scale(tiles["x"])
tiles["y_norm"] = minmax_scale(tiles["y"])

patients = tiles["patient_id"].unique()

count = 0
for p in patients:
    print(count/len(patients))
    is_selected = tiles["patient_id"] == p
    selected_tiles = tiles[is_selected][["x_norm", "y_norm", "p"]]
    dbscan = DBSCAN(eps=0.05, min_samples=5, n_jobs=-1)
    clusters = dbscan.fit_predict(selected_tiles.values)
    tiles.loc[is_selected, "cluster"] = clusters
    count += 1

    df = pd.pivot_table(tiles[is_selected], index="y", columns=["x"], values="p", fill_value=-1)
    
    plt.figure()
    sns.heatmap(df, vmin=-1, vmax=1., center=0, cmap="RdGy_r")
    plt.savefig(f"predict_patient_level_probas_{p}.png")
    plt.close()

    df = pd.pivot_table(tiles[is_selected], index="y", columns=["x"], values="cluster", fill_value=-2)
    plt.figure()
    sns.heatmap(df, vmin=-2, vmax=5, center=0, cmap="Set1")
    plt.savefig(f"predict_patient_level_cluster_{p}.png")
    plt.close()
    

#predict on real test set
is_isolated = tiles["cluster"] == -1
tiles = tiles[~is_isolated]
predictions = tiles.groupby("patient_id").max()[["p"]]
predictions.index = [str(i).zfill(3) for i in predictions.index]
predictions.index.name = "ID"
predictions.rename(columns={"p": "Target"}, inplace=True)
predictions.loc[TestPatients().ids()].to_csv("predict_patient_level.csv")
