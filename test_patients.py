import os
import glob

class TestPatients(object):
    """Patient test set"""
    
    PATH = "/Users/ldocao/Documents/Personnel/Recherche emploi/data science/2019 02 05 Owkin/interview 3/data/test_input"
    
    def __init__(self, predictions):
        """
        Parameters
        ----------
        predictions: pd.DataFrame{TARGET_COLUMN}
            patient id as index, and predicted presence of metastasis (1.0: yes, 0.0: no)
        """
        self.predictions = predictions


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

        

        
        
