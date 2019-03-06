# PURPOSE: predict random to test submission script

import numpy as np
import pandas as pd

from camelyon16 import TestPatients
from challenge import Challenge


TARGET = Challenge.TARGET_COLUMN
patient_ids = TestPatients().ids()
n_patients = len(patient_ids)

random_probabilities = np.random.uniform(size=n_patients)
Challenge(patient_ids, random_probabilities).submit("test_predictions.csv")

