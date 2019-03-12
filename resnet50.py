from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model


from image_classifier import ImageClassifier



class ResNet:
    INPUT_SHAPE = (96, 96, 3)

    def __init__(self):
        self.base_model = ResNet50(include_top=False,
                                   input_shape=self.INPUT_SHAPE,
                                   weights="imagenet",
                                   pooling=None)
    
    def transfer_learning(self):
        x = self.base_model.output
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out3 = Flatten()(x)
        out = Concatenate(axis=-1)([out1, out2, out3])
        out = Dropout(0.5)(out)
        out = Dense(1, activation="sigmoid", name="3_")(out)
        model = Model(self.base_model.input, out)

        for layer in self.base_model.layers:
            layer.trainable = False

        LEARNING_RATE = 0.000003
        model.compile(optimizer=Adam(LEARNING_RATE, decay=1e-6),
                      loss=binary_crossentropy,
                      metrics=['acc'])

        return model
