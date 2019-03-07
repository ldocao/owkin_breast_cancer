#PURPOSE: transfer learning from inception v3
import os
from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3

from inceptionv3 import InceptionV3, InceptionAlbum
from kaggle import TrainingImages


# data
SUB_SAMPLE = 2
image_filenames = TrainingImages().filenames[:SUB_SAMPLE]
ground_truths = TrainingImages().ground_truths.head(SUB_SAMPLE)

#format for fit
x, exceptions = InceptionAlbum(image_filenames).preprocess()
y = ground_truths["label"].values.T







# setup model
N_EPOCHS = 2
original = KerasInceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=(299, 299, 3),
                            pooling="avg")

inception = InceptionV3(original)
inception.transfer_learning()



if len(exceptions) == 0:
    inception.model.fit(x=x, y=y, batch_size=SUB_SAMPLE, epochs=N_EPOCHS)
