#PURPOSE: example of heatmap for a tumoral patient

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from camelyon16 import TrainingPatients

tiles = TrainingPatients().annotations
patients = tiles["ID"].unique()

for p in patients:
    PATIENT = p

    is_selected = tiles["ID"] == PATIENT
    selected_tiles = tiles[is_selected][["x", "y", "Target"]]

    df = pd.pivot_table(selected_tiles, values="Target", index="y", columns=["x"], fill_value=-1)

    plt.figure()
    sns.heatmap(df, vmin=-1, vmax=1.5, center=0, cmap="RdGy_r")
    plt.savefig(f"heatmap_tumor_{PATIENT}.png")
