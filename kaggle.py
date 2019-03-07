#PURPOSE: kaggle dataset

import cv2

class TiffImage:
    def __init__(self, path):
        self.path = path

    def open(self):
        bgr_img = cv2.imread(self.path) # bgr format by default
        b, g, r = cv2.split(bgr_img)
        rgb_img = cv2.merge([r, g, b])
        return rgb_img
        
