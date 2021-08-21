import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow.keras.backend as K

# +

batch_size = 16

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    rotation_range=10,
    shear_range=0.1,
    brightness_range=[0.4,1.25],
    horizontal_flip=True,
)


test_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=10,
    shear_range=0.1,
    brightness_range=[0.4,1.25],
    horizontal_flip=True,
)


train_generator = train_datagen.flow_from_directory(
        PATH_DATASET+'train',  
        target_size=(HEIGHT, WIDTH),  
        batch_size=batch_size,
        class_mode='binary') 

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        PATH_DATASET+'validation',
        target_size=(HEIGHT, WIDTH),
        batch_size=batch_size,
        class_mode='binary')
