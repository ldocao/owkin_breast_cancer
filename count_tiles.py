import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from camelyon16 import TrainingPatients, TestPatients

test = TestPatients().tiles
test["dataset"] = "test"
training = TrainingPatients().tiles
training["dataset"] = "training"

tiles = pd.concat([training, test])
n_tiles_per_patient = tiles.groupby("patient_id").count()["tile_id"]


#correlation matrix
corr = tiles.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1., vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.savefig("count_tiles_correlation.png")

#tiles per patient
plt.figure()
plt.hist(n_tiles_per_patient, bins=np.arange(0, 1100, 100), density=False)
plt.xlabel("number of tiles per patient")
plt.ylabel("number of patients")
plt.yscale("log")
plt.savefig("count_tiles_histogram.png")

#zoom distribution
plt.figure()
tiles.groupby(["zoom_level", "dataset"])["tile_id"].count().unstack("dataset").plot(kind="barh")
plt.xlabel("number of tiles")
plt.ylabel("zoom level")
plt.xscale("log")
plt.savefig("count_tiles_zoom.png") 



#correlation n_tiles, target
n_tiles = tiles.groupby("patient_id").count()[["tile_id"]]
gt = TrainingPatients().ground_truths
n_tiles.join(gt).corr()
