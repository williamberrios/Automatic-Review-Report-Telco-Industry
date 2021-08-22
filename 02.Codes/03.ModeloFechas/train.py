# +
import pandas as pd
import seaborn as sns
import pickle
import torch
import os
import time
import glob
from tqdm import tqdm
module_path = "../../src"
import pickle
import sys
from pt_soft_nms import soft_nms as snms
import config
if module_path not in sys.path:
    sys.path.append(module_path)
from fechas.dataset.dataset import DateDataset
from fechas.model.FasterRCNN import FasterRCNNTorchVision
from fechas.model.trainerFasterRCNN import *
from fechas.utils.seed import seed_everything
from fechas.utils.augmentations import get_transformations
from fechas.utils.plot import plot_date
from fechas.utils.preprocessing import get_date_from_prediction
pd.set_option('max_rows',999)

# ======= GENERAL PARAMETERS ========
DATA_PATH = '../../01.Datasets'
MEAN = [0,0,0]
STD  = [1,1,1]


# -

def train():
    print("========================================")
    print("=========Training Modelo Fecha==========")
    print("========================================")
    data_dict   = open(os.path.join(DATA_PATH,"data_dict.pkl"), "rb")
    data_dict   = pickle.load(data_dict)
    df_train  = pd.read_csv(os.path.join(DATA_PATH,"train_date_detection.csv"))
    print('Dictionary:\n',data_dict)
    seed_everything(config.seed)
    model = FasterRCNNTorchVision(config.name_model,
                                  config.num_classes,
                                  pretrained_coco = config.pretrained_coco,
                                  pretrained_back = config.pretrained_back).ReturnModel()
    
    trainer = TrainerFasterRCNN(config = config,
                                model  = model,
                                dict_transforms = get_transformations(MEAN,STD))
    # Because there is not soo much training data, we train with the whole data and perform early stopping
    train = df_train.reset_index(drop = True)
    valid = df_train.reset_index(drop = True)
    ret  = trainer.fit(train = train,valid = valid,output_path= config.output_path)


if __name__ == '__main__':
    train()

'''
def predict(test,config,image_dir_test,classes,nms = 0.2):
    # Generate Model
    model = FasterRCNNTorchVision(config.name_model,
                                  config.num_classes,
                                  pretrained_coco = config.pretrained_coco,
                                  pretrained_back = config.pretrained_back).ReturnModel()
    model.load_state_dict(torch.load(config.output_path))
    # Generate Trainer
    trainer = TrainerFasterRCNN(config = config,
                                model  = model,
                                dict_transforms = get_transformations(MEAN,STD))
    
    # Predict for Dataset:
    df = trainer.predict_nms(test = test,image_dir_test = image_dir_test,classes = data_dict, nms = 0.2)
    df = get_date_from_prediction(df)
    return df
'''

# +
#data_dict   = open(os.path.join(DATA_PATH,"data_dict.pkl"), "rb")
#data_dict   = pickle.load(data_dict)
#df_test     = pd.read_csv(os.path.join(DATA_PATH,"test_date_detection.csv"))
#df_test     = predict(df_test,config,'',data_dict,nms = 0.2)
