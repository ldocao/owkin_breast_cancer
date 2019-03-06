import pandas as pd

class Challenge(object):
    """Submission script"""
    N_DIGITS = 3 #number of digits for patient id
    TARGET_COLUMN = "Target"
    INDEX_NAME = "ID"

    def __init__(self, patient_ids, probabilities):
        self.patient_ids = patient_ids
        self.probabilities = probabilities
        self.predictions = self._format_output()
        

    def submit(self, filename):
        self._check_format()
        self.predictions.to_csv(filename)
        

    def _format_output(self):
        cls = self.__class__
        predictions = pd.DataFrame({cls.TARGET_COLUMN: self.probabilities},
                                   index=self.patient_ids)
        return predictions

    
    def _check_format(self):
        self._rename_and_type_columns()
        self._check_target_range()
            
    def _rename_and_type_columns(self):
        cls = self.__class__
        
        self.predictions.index.name = cls.INDEX_NAME
        self.predictions.index = [str(i).zfill(cls.N_DIGITS) for i in self.predictions.index]
        
        self.predictions.index = self.predictions.index.astype(str)
        self.predictions[cls.TARGET_COLUMN] = self.predictions[cls.TARGET_COLUMN].astype(float)

    def _check_target_range(self):
        cls = self.__class__
        assert 0 <= self.predictions[cls.TARGET_COLUMN].min() <= 1.
        assert 0 <= self.predictions[cls.TARGET_COLUMN].max() <= 1.
        

