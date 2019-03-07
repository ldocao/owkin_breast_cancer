#PURPOSE: kaggle dataset

import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

class TiffImage:
    def __init__(self, path):
        if isinstance(path, Path):
            path = str(path.resolve())
        self.path = path

    def open(self):
        """Returns image content as numpy array"""
        bgr_img = cv2.imread(self.path) # bgr format by default
        b, g, r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r, g, b])
        return rgb_img
        

class Kaggle:
    ROOT_PATH = "/Users/ldocao/Documents/Personnel/Recherche emploi/data science/2019 02 05 Owkin/interview 3/kaggle"

    
    @property
    def filenames(self):
        return sorted(self.path.glob("*.tif"))
    

    
class TrainingImages(Kaggle):
    GROUND_TRUTH = "train_labels.csv"
    TRAINING_SIZE = 0.8 
    
    def __init__(self, path="train"):
        self.path = Path(self.__class__.ROOT_PATH) / path

        
    @property
    def ground_truths(self):
        cls = self.__class__
        ground_truths = pd.read_csv(self.path / cls.GROUND_TRUTH)
        ground_truths.set_index("id", inplace=True)
        return ground_truths.sort_index()

    @property
    def training(self):
        SEED = 1
        np.random.seed(SEED)
        all_paths = self.ground_truths.index.values
        n_samples = int(len(all_paths)*self.__class__.TRAINING_SIZE)
        return np.random.choice(all_paths, n_samples, replace=False) #gt is balanced


    @property
    def validation(self):
        remaining = set(self.ground_truths.index) - set(self.training)
        return remaining


class TestImages(Kaggle):
    
    def __init__(self, path="test"):
        self.path = Path(self.__class__.ROOT_PATH) / path

   
