# +
import pandas as pd
import numpy as np
import sys
import torch
import cv2
module_path = "../../../src"
if module_path not in sys.path:
    sys.path.append(module_path)

from fechas.model.trainerFasterRCNN import TrainerFasterRCNN
from fechas.model.FasterRCNN import FasterRCNNTorchVision
from fechas.utils.augmentations import get_transformations
from fechas.utils.preprocessing import get_date_from_prediction


# -

class fecha_model:
    def __init__(self,config,data_dict):
        self.config = config
        self.base_model = self._getmodel()
        self.trainer = self._gettrainer()
        self.data_dict = data_dict
        
        
    def _getmodel(self):
        model = FasterRCNNTorchVision(self.config.name_model,
                                      self.config.num_classes,
                                      pretrained_coco = self.config.pretrained_coco,
                                      pretrained_back = self.config.pretrained_back).ReturnModel()
        model.to(self.config.device)
        model.load_state_dict(torch.load(self.config.output_path))
        return model
                              
        
                              
    def _gettrainer(self):
        return TrainerFasterRCNN(config = self.config,
                                 model  = self.base_model,
                                 dict_transforms = get_transformations([0,0,0],[1,1,1]))
                              
    def predict_example(self,img,path_aux = './img.jpg'):
        # save image in path
        cv2.imwrite(path_aux,img)
        aux = pd.DataFrame({'filename':['img_predict'],'filename_path':[path_aux]})                
        aux = self.trainer.predict_nms(test = aux,image_dir_test = '',classes = self.data_dict, nms = 0.2)
        return  get_date_from_prediction(aux)[['date_day','date_month','date_year']].values
