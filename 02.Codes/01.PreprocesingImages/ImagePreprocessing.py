import os
import sys
import pandas as pd
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from detector.detector import detector
from detector.utils import processing_images
DATA_PATH = '../../01.Datasets'


def main():
    df_train = pd.read_csv(os.path.join(DATA_PATH,'files','output_train.csv'))
    df_test  = pd.read_csv(os.path.join(DATA_PATH,'files','sampleSubmission.csv'))
    img_reference_path  = os.path.join(DATA_PATH,'files/reference.png')
    my_detector         = detector(filename = img_reference_path)
    # Processing Images Train:
    processing_images(df_train,my_detector,DATA_PATH,'images_train')
    processing_images(df_test,my_detector,DATA_PATH,'images_test')


if __name__ == '__main__':
    main()


