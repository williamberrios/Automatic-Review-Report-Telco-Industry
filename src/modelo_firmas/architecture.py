import os
import numpy as np
from .utils import focal_loss
from .utils import custom_f1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121


def modelo_firmas(WIDTH,HEIGHT):

    base_model = DenseNet121(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH,3))

    model = tf.keras.models.Sequential()

    model.add(base_model)

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.layers[0].trainable = False

    model.compile(
        loss =[focal_loss],
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.AUC(),'accuracy',custom_f1]

    )

    return model