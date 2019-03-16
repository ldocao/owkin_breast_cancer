import numpy as np

class TileFeature:
    def __init__(self, features):
        self.features = features

    def engineer(self):
        QUARTILES = [25, 50, 75]
        x1 = np.mean(self.features)
        x2 = np.std(self.features)
        x3 = np.percentile(self.features, QUARTILES[0])
        x4 = np.percentile(self.features, QUARTILES[1])
        x5 = np.percentile(self.features, QUARTILES[2])
        x6 = np.max(self.features)
        x7 = np.min(self.features)
        x8 = (self.features > 0.5).sum() / len(self.features)
        x9 = len(self.features) #number of tiles
        return (x1, x2, x3, x4, x5, x6, x7, x8, x9)

