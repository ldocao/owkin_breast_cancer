from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.applications.nasnet import NASNetMobile
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model


from image_classifier import ImageClassifier





class NasnetMobile:
    INPUT_SHAPE = (96, 96, 3)

    def __init__(self):
        self.base_model = NASNetMobile(include_top=False,
                                       input_shape=self.INPUT_SHAPE,
                                       weights="imagenet")
    
    def transfer_learning(self):
        LEARNING_RATE = 0.001 
        inputs = Input(self.INPUT_SHAPE)
        x = self.base_model(inputs)
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out3 = Flatten()(x)
        out = Concatenate(axis=-1)([out1, out2, out3])
        out = Dropout(0.5)(out)
        out = Dense(1, activation="softmax", name="dense_3")(out)
        model = Model(inputs, out)

        for layer in self.base_model.layers:
            layer.trainable = False
    
        model.compile(optimizer=Adam(LEARNING_RATE, decay=1e-6),
                      loss=binary_crossentropy,
                      metrics=['acc'])
        print(model.summary())
        return model
