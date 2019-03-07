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
    PREFIX = "ID_"


    
    def resnet_features(self, patient_id):
        """Returns resnet features (without location)

        Parameters
        ----------
        patient_id: str or int
            patient identifier

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

    
    @classmethod
    def _id_to_filename(cls, patient_id):
        """Returns prefixed name of file from patient_id

        Parameters
        ----------
        patient_id: str or int
            patient identifier 
        """
        return cls.PREFIX + str(patient_id).zfill(Challenge.N_DIGITS)



    

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
    ANNOTATION = "train_tile_annotations.csv"
    
    def __init__(self, path="train_input"):
        """
        Parameters
        ----------
        path: str
            path from main Camelyon16 dataset directory
        """
        self.path = os.path.join(self.ROOT_PATH, path)

    @property
    def ground_truths(self):
        """Returns ground truth labels as dataframe"""
        ground_truths = pd.read_csv(self.ground_truth_path())
        ground_truths.set_index(Challenge.INDEX_NAME, inplace=True)
        return ground_truths

    @property
    def annotations(self):
        """Returns splitted information on annotated tiles"""
        cls = self.__class__
        path = Path(self.path) / cls.ANNOTATION
        FILENAME = "filename"
        columns = [FILENAME, Challenge.TARGET_COLUMN]
        annotations = pd.read_csv(path, names=columns, skiprows=1)
        separate_info = lambda row: pd.Series(row.split("_"))
        tile_infos = annotations[FILENAME].apply(separate_info)
        tile_infos[7] = tile_infos[7].apply(lambda row: row.split(".")[0])
        tile_infos.drop([0, 2, 3], axis=1, inplace=True)
        tile_infos.rename(columns={1: Challenge.INDEX_NAME,
                                   4: "tile_id",
                                   5: "zoom_level",
                                   6: "x",
                                   7: "y"}, inplace=True)
        tile_infos.index = annotations[FILENAME]
        int_columns = ["zoom_level", "x", "y"]
        tile_infos[int_columns] = tile_infos[["zoom_level", "x", "y"]].astype(int)
        return tile_infos


    @classmethod
    def ground_truth_path(cls):
        return os.path.join(cls.ROOT_PATH, cls.GROUND_TRUTH)
