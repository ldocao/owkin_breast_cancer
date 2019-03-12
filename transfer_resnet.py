import ipdb

import os
import cv2
import json
from pathlib import Path

import numpy as np
import pandas as pd

import keras
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from challenge import Challenge
from camelyon16 import TrainingPatients, AnnotatedTile
from resnet50 import ResNet


#hardware specs
BATCH_SIZE = 32
ANNOTATED_TILES_PATH = "/home/ldocao/owkin/data/train_input/annotated_tiles"

#model
resnet = ResNet().transfer_learning()
print(resnet.summary())

# raw data
print("load data")
training, validation = TrainingPatients().split_tiles_train_validation()
tiles = TrainingPatients().annotations



# # data augmentation
print("define data augmentation")

train_df = tiles.loc[training]
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_dataframe(train_df.reset_index(),
                                                    directory=ANNOTATED_TILES_PATH,
                                                    x_col=tiles.index.name, y_col=Challenge.TARGET_COLUMN,
                                                    target_size=ResNet.INPUT_SHAPE[:2],
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')

validation_df = tiles.loc[validation]
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_dataframe(validation_df.reset_index(),
                                                              directory=ANNOTATED_TILES_PATH,
                                                              x_col=tiles.index.name, y_col=Challenge.TARGET_COLUMN,
                                                              target_size=ResNet.INPUT_SHAPE[:2],
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary')




#auto numbering
TENSORBOARD_PATH = Path("/home/ldocao/owkin/tensorboard")
run_identifier = 0

run_dir_exists = True
while run_dir_exists:
    run_identifier += 1
    run_id = str(run_identifier).zfill(3)
    new_dir = TENSORBOARD_PATH / f"run_{run_id}"
    run_dir_exists = new_dir.exists()

else:
    new_dir.mkdir()
    print("output path:", new_dir)

#checkpoint strategy
print("define checkpoint strategy")
filepath = new_dir / "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(str(filepath),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')


tbCallBack = keras.callbacks.TensorBoard(log_dir=str(new_dir),
                                         histogram_freq=0,
                                         update_freq="batch",
                                         write_graph=True,
                                         write_images=True)
callbacks_list = [tbCallBack, checkpoint]

# train the model
print("train")
N_STEPS_PER_EPOCH = len(training) // BATCH_SIZE
VALIDATION_STEPS = len(validation) // BATCH_SIZE
N_EPOCHS = 20
history = resnet.fit_generator(train_generator,
                               steps_per_epoch=N_STEPS_PER_EPOCH,
                               epochs=N_EPOCHS,
                               validation_data=validation_generator,
                               validation_steps=VALIDATION_STEPS,
                               use_multiprocessing=True,
                               callbacks=callbacks_list)

with open('transfer_nasnet_mobile.json', 'w') as fp:
        json.dump(history.history, fp)

