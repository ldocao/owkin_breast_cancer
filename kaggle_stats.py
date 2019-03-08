import pandas as pd

from kaggle import TrainingImages, TestImages

gt = TrainingImages().ground_truths
print("n training", len(gt))
print("n test", len(TestImages().filenames))
print(gt["label"].value_counts())

