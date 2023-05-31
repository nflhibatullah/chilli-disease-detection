import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

def make_model():

    mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3),
                                                          include_top=False,
                                                          weights='imagenet')

    for layer in mobilenet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(mobilenet)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(8192, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(5, activation="softmax", name="classification"))
    
    return model
