"""Module containing preprocessing tools"""

import dask
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, Dropout, Flatten

from image_classifier import ImageClassifier





class InceptionV3(ImageClassifier):
    
    def transfer_learning(self):
        self.model = self._add_new_last_layer()
        self._freeze_layers()

    def _add_new_last_layer(self):
        cls = self.__class__
        
        x = self.base_model.output
        predictions = Dense(1, activation='softmax')(x) #new softmax layer
        model = Model(input=self.base_model.input, output=predictions)
        return model

    def _freeze_layers(self):
        for layer in self.base_model.layers:
            layer.trainable = False
            
        self.model.compile(optimizer='rmsprop',
                           loss='binary7_crossentropy',
                           metrics=['accuracy'])




class InceptionImage(object):
    INPUT_SIZE = (299, 299)

    def __init__(self, path):
        self.path = path
        self.content = self._open()

    def preprocess(self):
        """Returns content of image normalized for inception model"""
        x = self.content
        x /= 255.
        x -= 0.5
        x *= 2.
        x = np.expand_dims(x, axis=0)
        return x


    def _open(self):
        """Returns content of image as numpy array and resized"""
        content = image.load_img(self.path, target_size=self.INPUT_SIZE)
        content = image.img_to_array(content)
        return content


    
class InceptionAlbum(object):
    def __init__(self, paths):
        """
        Parameters
        ----------
        paths: iterable
            path to image to load
        """
        self.paths = paths

    def preprocess(self):
        """Returns preprocessed images as np.array
        
        Returns
        -------
        pixels: np.array
            content of images as numpy array (n_images, 299, 299, 3)
        """
        self.pixels, index_exceptions = preprocess_in_parallel(self.paths)
        return self.pixels, index_exceptions
        

    

@dask.delayed
def _load(s):
    return s


@dask.delayed
def _preprocess(path):
    """Returns preprocessed image for Inception v3 or path if this raises an OSError

    Parameters
    ----------
    path: str
        path to image

    Returns
    -------
    content: np.array
        pixel values of the image. Returns a matrix full of NaN if the image cannot be loaded
    """
    try:
        return InceptionImage(path).preprocess()
    except OSError:
        print("{} cannot be loaded. Returning None".format(path))
        LOADING_ERROR = None #cannot choose np.nan since it is converted to None by dask
        return LOADING_ERROR


def preprocess_in_parallel(paths):
    """Returns an array of preprocessed images
    
    Parameters
    ----------
    paths: list of str
        path of images to preprocess

    Returns
    -------
    contents: np.array(n, 299, 299, 3)
        contents of all preprocessed images. If an image has not been loaded, a matrix full of NaN is returned
    """
    data = [_load(p) for p in paths]
    contents = [_preprocess(d) for d in data]
    contents = dask.compute(contents)[0] #returns a tuple of 1-element
    # contents = [c[0] if c is not None else None for c in contents]
    # index_exceptions = [contents.index(c) for c in contents if c is None]
    index_exceptions = list()

    for i in range(len(contents)):
        if contents[i] is None:
            index_exceptions.append(i)
    contents = [c[0] for c in contents if c is not None]
    return np.array(contents), index_exceptions

