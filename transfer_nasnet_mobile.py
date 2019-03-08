from keras.callbacks import ModelCheckpoint
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from kaggle import TiffImage, TrainingImages, TestImages
from nasnet_mobile import NasnetMobile


#hardware specs
BATCH_SIZE = 32


# raw data
print("load data")
training = TrainingImages().training
validation = TrainingImages().validation
gt = TrainingImages().ground_truths
test = TestImages().filenames



# data augmentation
print("define data augmentation")
train_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_input)

train_df = gt.loc[training]
train_df.index = train_df.index + ".tif"
train_df.index.name = "id"
directory = TrainingImages().path
train_generator = train_datagen.flow_from_dataframe(train_df.reset_index(),
                                                    directory,
                                                    x_col="id", y_col="label", 
                                                    target_size=NasnetMobile.INPUT_SHAPE[:2],
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary')

validation_df = gt.loc[validation]
validation_df.index = validation_df.index + ".tif"
validation_df.index.name = "id"
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_dataframe(validation_df.reset_index(),
                                                              directory,
                                                              x_col="id", y_col="label", 
                                                              target_size=NasnetMobile.INPUT_SHAPE[:2],
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='binary')
            


# model
print("transfer learning")
model = NasnetMobile().transfer_learning()



#checkpoint strategy
print("define checkpoint strategy")
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]


# train the model
print("train")
N_STEPS_PER_EPOCH = len(training) // BATCH_SIZE
VALIDATION_STEPS = len(validation) // BATCH_SIZE
N_EPOCHS = 10
history = model.fit_generator(train_generator,
                              steps_per_epoch=N_STEPS_PER_EPOCH,
                              epochs=N_EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=VALIDATION_STEPS,
                              use_multiprocessing=True,
                              callbacks=callbacks_list)

with open('transfer_nasnet_mobile.json', 'w') as fp:
    json.dump(history.history, fp)
