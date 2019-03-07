#PURPOSE: estimate how many patients are positive but not annotated

from camelyon16 import TrainingPatients
from challenge import Challenge

annotations = TrainingPatients().annotations
n_annotated_patients = len(annotations[Challenge.INDEX_NAME].unique())
print("number of annotated patients", n_annotated_patients)

ground_truths = TrainingPatients().ground_truths
is_tumoral = ground_truths[Challenge.TARGET_COLUMN] == 1
tumoral_patients = ground_truths[is_tumoral].index
print("number of tumoral patients", len(tumoral_patients))
