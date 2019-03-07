#PURPOSE: transfer learning from inception v3
import ipdb
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns

from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.preprocessing.image import ImageDataGenerator

from inceptionv3 import InceptionV3, InceptionImage, InceptionAlbum
from kaggle import TrainingImages


# raw data
print("loading raw data")
image_filenames = TrainingImages().filenames
#X_train, exceptions = InceptionAlbum(image_filenames).preprocess()
gt = TrainingImages().ground_truths
Y_train = gt["label"].values.T



# data augmentation
print("define data augmentation")
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=InceptionImage._preprocess)

train_df = gt.loc[TrainingImages().training]
train_df.index = train_df.index + ".tif"
train_df.index.name = "id"
directory = TrainingImages().path
train_generator = train_datagen.flow_from_dataframe(train_df.reset_index(),
                                                    directory,
                                                    x_col="id", y_col="label", 
                                                    target_size=InceptionImage.INPUT_SIZE,
                                                    batch_size=32,
                                                    class_mode='binary')

validation_df = gt.loc[TrainingImages().validation]
validation_df.index = validation_df.index + ".tif"
validation_df.index.name = "id"
validation_datagen = ImageDataGenerator(preprocessing_function=InceptionImage._preprocess,)
validation_generator = validation_datagen.flow_from_dataframe(validation_df.reset_index(),
                                                              directory,
                                                              x_col="id", y_col="label", 
                                                              target_size=InceptionImage.INPUT_SIZE,
                                                              batch_size=32,
                                                              class_mode='binary')


# setup model
print("transfer learning")
N_EPOCHS = 2
original = KerasInceptionV3(include_top=False,
                            weights='imagenet',
                            input_shape=(299, 299, 3),
                            pooling="avg")

inception = InceptionV3(original)
inception.transfer_learning()


# train the model
print("train")
N_STEPS_PER_EPOCH = 200
N_EPOCHS = 50
VALIDATION_STEPS = 50
history = inception.model.fit_generator(train_generator,
                                        steps_per_epoch=N_STEPS_PER_EPOCH,
                                        epochs=N_EPOCHS,
                                        validation_data=validation_generator,
                                        validation_steps=VALIDATION_STEPS,
                                        use_multiprocessing=True)

with open('predict_inceptionv3.json', 'w') as fp:
    json.dump(history.history, fp)

# Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
