import abc


class ImageClassifier:
    N_CLASSES = 2 #tumoral or not
    
    def __init__(self, base_model):
        self.base_model = base_model
        

    @abc.abstractmethod
    def transfer_learning(self):
        raise NotImplementedError



    
