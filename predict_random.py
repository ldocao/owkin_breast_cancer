# PURPOSE: predict random to test submission script

import numpy as np
import pandas as pd

from test_patients import TestPatients


TARGET = TestPatients.TARGET_COLUMN
patient_ids = TestPatients.ids()
n_patients = len(patient_ids)

random_probabilities = np.random.uniform(size=n_patients)
predictions = pd.DataFrame({TARGET: random_probabilities}, index=patient_ids)

TestPatients(predictions).submit("test_predictions.csv")
