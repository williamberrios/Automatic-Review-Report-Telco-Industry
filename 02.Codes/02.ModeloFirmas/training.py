import os
import sys
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
    
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import  EarlyStopping,ModelCheckpoint

import tensorflow as tf

from modelo_firmas.utils import seed_everything
from modelo_firmas.utils import custom_f1
from modelo_firmas.utils import plot_history
from modelo_firmas.architecture import modelo_firmas
from modelo_firmas.evaluate import evaluate_model







WIDTH = 350
HEIGHT = 100

PATH_DATASET = '../../01.Datasets/processed/images_train/firmas_modelo/modelamiento/'
PATH_RESULTS = '../../03.SavedModels'

MODEL_NAME = 'modelo_firmas.h5'
BATCH_SIZE = 16

NUM_EPOCHS = 1000

def main():

    seed_everything(42)
    
    # Load model architecture
    model = modelo_firmas(WIDTH,HEIGHT)
    model.summary()

    # Load image data generator for train
    train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=10,
        shear_range=0.1,
        brightness_range=[0.4,1.25],
        horizontal_flip=True,
    )

    # Load image data generator for validation
    val_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=10,
        shear_range=0.1,
        brightness_range=[0.4,1.25],
        horizontal_flip=True,
    )

    # Load train dataset
    train_generator = train_datagen.flow_from_directory(
            PATH_DATASET+'train',  # this is the target directory
            target_size=(HEIGHT, WIDTH),  # all images will be resized to (HEIGHT, WIDTH)
            batch_size=BATCH_SIZE,
            class_mode='binary')  #  we need binary labels

    # Load validation dataset
    validation_generator = val_datagen.flow_from_directory(
            PATH_DATASET+'validation',
            target_size=(HEIGHT, WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary')
    
    # Earlystopping
    earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss',mode='min')

    # Training
    history = model.fit_generator(train_generator,steps_per_epoch= len(train_generator),epochs=NUM_EPOCHS,
        validation_data=validation_generator,validation_steps= len(validation_generator),callbacks=[earlyStopping, mcp_save])
    
    model.save(os.path.join(PATH_RESULTS,MODEL_NAME))
    
    fig_model_performance = plot_history(history)
    fig_model_performance.savefig("graficos/metricas de training.jpg")

if __name__ == '__main__':
    main()
