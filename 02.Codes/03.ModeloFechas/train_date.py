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
pd.set_option('max_rows',999)

# ======= GENERAL PARAMETERS ========
DATA_PATH = '../../01.Datasets'
MEAN = [0,0,0]
STD  = [1,1,1]
# -

data_dict   = open(os.path.join(DATA_PATH,"data_dict.pkl"), "rb")
data_dict   = pickle.load(data_dict)
num_classes = len(data_dict) +1
df_train  = pd.read_csv(os.path.join(DATA_PATH,"train_date_detection.csv"))
print('Dictionary:\n',data_dict)


def main():
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


# %%time
main()


def get_predictions_fastercnn(df,config,image_dir_test = ''):
    # Generate Model
    model = FasterRCNNTorchVision(config.name_model,
                                  config.num_classes,
                                  pretrained_coco = config.pretrained_coco,
                                  pretrained_back = config.pretrained_back).ReturnModel()
    model.load_state_dict(torch.load(config.output_path))
    # Generate Trainer
    trainer = TrainerFasterRCNN(config = config,
                                model  = model,
                                dict_transforms = get_transformations())
    # Predict for Dataset:
    return trainer.predict(test = df,image_dir_test = image_dir_test)


df_test.head

l
