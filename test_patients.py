import os
import glob

class TestPatients:
    """Patient test set"""
    
    PATH = "/Users/ldocao/Documents/Personnel/Recherche emploi/data science/2019 02 05 Owkin/interview 3/data/test_input"
    N_DIGITS = 3 #number of digits for patient id
    TARGET_COLUMN = "Target"
    INDEX_NAME = "ID"
    
    def __init__(self, predictions):
        """
        Parameters
        ----------
        predictions: pd.DataFrame{TARGET_COLUMN}
            patient id as index, and predicted presence of metastasis (1.0: yes, 0.0: no)
        """
        self.predictions = predictions

    def submit(self, filename):
        self._check_format()
        self.predictions.to_csv(filename)


    @classmethod
    def ids(cls):
        """Return ids of patients

        Returns
        -------
            numbers: ['004', '338', ...]
        """
        SUB_FOLDER = "resnet_features"
        path = os.path.join(cls.PATH, SUB_FOLDER)
        absolute_paths = glob.glob(path+"/*.npy")
        basenames = [os.path.basename(s) for s in absolute_paths]
        ids = [s.split(".")[0] for s in basenames]
        numbers = [s.split("_")[1] for s in ids]
        return sorted(numbers)

        
    def _check_format(self):
        self._rename_and_type_columns()
        self._check_target_range()
        
        
    def _rename_and_type_columns(self):
        self.predictions.index.name = self.INDEX_NAME
        self.predictions.index = [str(i).zfill(self.N_DIGITS) for i in self.predictions.index]
        
        self.predictions.index = self.predictions.index.astype(str)
        self.predictions[self.TARGET_COLUMN] = self.predictions[self.TARGET_COLUMN].astype(float)


    def _check_target_range(self):
        assert 0 <= self.predictions[self.TARGET_COLUMN].min()
        assert 1 >= self.predictions[self.TARGET_COLUMN].max()
        


        
        
