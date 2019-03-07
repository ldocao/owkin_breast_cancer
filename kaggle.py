#PURPOSE: kaggle dataset

from pathlib import Path

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
        return self.path.glob("*.tif")
    

    
class TrainingImages(Kaggle):
    GROUND_TRUTH = "train_labels.csv"
    
    def __init__(self, path="train"):
        self.path = Path(self.__class__.ROOT_PATH) / path

        
    def ground_truths(self):
        cls = self.__class__
        ground_truths = pd.read_csv(self.path / cls.GROUND_TRUTH)
        ground_truths.set_index("id", inplace=True)
        return ground_truths




class TestImages(Kaggle):
    
    def __init__(self, path="test"):
        self.path = Path(self.__class__.ROOT_PATH) / path

   
