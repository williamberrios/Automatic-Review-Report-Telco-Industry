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


def main():
    # Evaluation
    evaluate_model(os.path.join(PATH_RESULTS,MODEL_NAME),PATH_DATASET,WIDTH,HEIGHT)


if __name__ == '__main__':
    main()
    
    

