__all__ = ["TrainingPatients", "TestPatients"]

import os
import glob
from pathlib import Path


import numpy as np
import pandas as pd

from challenge import Challenge





class Camelyon16(object):
    ROOT_PATH = "/home/ldocao/owkin/data"
    PREFIX = "ID_"

    def __init__(self, path=None):
        self.path = os.path.join(self.ROOT_PATH, path)

    @property
    def filenames(self):
        path = Path(self.path) / "images"
        files = path.glob("**/*.jpg")
        return files

    @property
    def tiles(self):
        COLUMNS = ["patient_id", "tile_id", "zoom_level", "x", "y", "is_annotated"]
        INDEX_NAME = "filename"
        tile_path = Path(self.path) / self.__class__.TILE_INFO
        if tile_path.exists():
            tile_infos = pd.read_csv(tile_path)
            tile_infos.set_index(INDEX_NAME, inplace=True)
        else:
            tile_infos = self._extract_tile_infos()
            tile_infos = pd.DataFrame.from_dict(tile_infos, orient="index", columns=COLUMNS)
            tile_infos.index.name = INDEX_NAME
            tile_infos.to_csv(tile_path)
        return tile_infos

    


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


    def _extract_tile_infos(self):
        """Returns tile informations"""
        tiles = Path(self.path).glob("images/**/*.jpg")
        tile_infos = {} #use dict structure to speed up for loop below
        for t in tiles:
            new_index = str(t.absolute()) #use absolute path as index
            tile_infos[new_index] = self.split_filename_info(t.name)
        return tile_infos
    
    @staticmethod
    def split_filename_info(filename):
        """Returns (patient_id, tile_id, zoom_level, x, y, is_annotated)"""
        separated_info = str(filename).split("_")
        is_annotated = len(separated_info) == 8
        separated_info[-1] = separated_info[-1].split(".")[0] #remove extension
        del separated_info[0]
        if is_annotated:
            del separated_info[1]
            del separated_info[1]
        else:
            del separated_info[1]
        separated_info[1:] = [int(s) for s in separated_info[1:]]
        separated_info.append(is_annotated)
        return separated_info

    
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
    TILE_INFO = "tile_informations.csv"
    
    def __init__(self, path="test_input"):
        """
        Parameters
        ----------
        path: str
            path from main Camelyon16 dataset directory
        """
        super().__init__(path)

        
        
class TrainingPatients(Camelyon16):
    GROUND_TRUTH = "training_output_bis_EyawEvU.csv"
    ANNOTATION = "train_tile_annotations.csv"
    TILE_INFO = "tile_informations.csv"
    
    def __init__(self, path="train_input"):
        """
        Parameters
        ----------
        path: str
            path from main Camelyon16 dataset directory
        """
        super().__init__(path)




        
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
        tile_infos[Challenge.TARGET_COLUMN] = annotations.set_index(FILENAME)[Challenge.TARGET_COLUMN]
        return tile_infos


    @classmethod
    def ground_truth_path(cls):
        return os.path.join(cls.ROOT_PATH, "train_input", cls.GROUND_TRUTH)
        





class AnnotatedTile:
    ROOT_PATH = Camelyon16.ROOT_PATH
    
    def __init__(self, filename):
        self.filename = filename

    @property
    def path(self):
        PATH = Path(self.__class__.ROOT_PATH) / "train_input/images" 
        directory = self.filename.split("_")[:3]
        directory = "_".join(directory)
        return PATH / directory / self.filename

        
