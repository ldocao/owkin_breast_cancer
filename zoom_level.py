#PURPOSE: exploratory on zoom levels
import itertools
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from camelyon16 import TrainingPatients

training_patients = TrainingPatients()
image_paths = Path(training_patients.path) / "images"
patients = image_paths.glob("*/")

# gather zoom levels for all tiles
zoom = []
for p in patients:
    print(p)
    tiles = p.glob("*.jpg")
    zoom_levels = [int(s.stem.split("_")[4]) for s in tiles]
    zoom.append(zoom_levels)


# count zoom levels
flat_zoom = np.array(list(itertools.chain(*zoom)))
counter = collections.Counter(flat_zoom)
df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
df.columns = ["zoom_level", "n_tiles"]
df.set_index("zoom_level", inplace=True)
df = df.sort_values("n_tiles", ascending=False)


df.head(5).plot(kind="barh")
plt.xlabel("count")
plt.ylabel("zoom level")
plt.savefig("zoom_level.png")
