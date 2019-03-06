__all__ = ["TrainingPatients", "TestPatients"]

import os
import glob
import abc
from pathlib import Path

import numpy as np
import pandas as pd

from challenge import Challenge




class Camelyon16(object):
    ROOT_PATH = "/Users/ldocao/Documents/Personnel/Recherche emploi/data science/2019 02 05 Owkin/interview 3/data"


    def resnet_features(self, patient_id):
        """Returns resnet features (without location)

        Parameters
        ----------
        patient_id: str
            patient identifier padded with Challenge.N_DIGITS leading zeros

        """
        EXTENSION = ".npy"
        SUBFOLDER = "resnet_features"
        basename = self._id_to_filename(patient_id)

        try: #is the file annotated?
            ANNOTATION = "_annotated"
            filename = basename + ANNOTATION + EXTENSION
            path = os.path.join(self.path, SUBFOLDER, filename)            
            features = np.load(path)
        except FileNotFoundError: #load default filename
            filename = basename + EXTENSION
            path = os.path.join(self.path, SUBFOLDER, filename)  
            features = np.load(path)
        finally:
            features = features[:, 3:] #remove location features
            return features


    def ids(self):
        """Return ids of patients

        Returns
        -------
            numbers: ['004', '338', ...]
        """
        absolute_paths = self._resnet_files()
        ids = [Path(s).stem for s in absolute_paths]
        numbers = [s.split("_")[1] for s in ids]
        return sorted(numbers)


    def _resnet_files(self):
        SUB_FOLDER = "resnet_features"
        path = Path(self.path) / SUB_FOLDER
        absolute_paths = path.glob("*.npy")
        return absolute_paths


    def _id_to_filename(self, patient_id):
        """Returns prefixed name of file from patient_id

        Parameters
        ----------
        patient_id: str
            patient identifier padded with Challenge.N_DIGITS leading zeros
        """
        PREFIX = "ID_"
        return PREFIX + patient_id



    

class TestPatients(Camelyon16):
    """Patient test set"""
    
    def __init__(self, path="test_input"):
        """
        Parameters
        ----------
        path: str
            path from main Camelyon16 dataset directory
        """
        self.path = os.path.join(self.ROOT_PATH, path)



        
        
class TrainingPatients(Camelyon16):
    GROUND_TRUTH = "training_output_bis_EyawEvU.csv"
    
    def __init__(self, path="train_input"):
        """
        Parameters
        ----------
        path: str
            path from main Camelyon16 dataset directory
        """
        self.path = os.path.join(self.ROOT_PATH, path)


    def ground_truths(self):
        """Returns ground truth labels as dataframe"""
        ground_truths = pd.read_csv(self.ground_truth_path())
        ground_truths.set_index(Challenge.INDEX_NAME, inplace=True)
        return ground_truths
        

    @classmethod
    def ground_truth_path(cls):
        return os.path.join(cls.ROOT_PATH, cls.GROUND_TRUTH)
