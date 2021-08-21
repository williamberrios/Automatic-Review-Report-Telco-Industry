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
df_test   = pd.read_csv(os.path.join(DATA_PATH,"test_date_detection.csv"))
print('Dictionary:\n',data_dict)

df_test['filename_path'][0]


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
#main()

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
                                dict_transforms = get_transformations(MEAN,STD))
    # Predict for Dataset:
    return trainer.predict(test = df,image_dir_test = image_dir_test)


df_test.head()

config2 = config
config2.output_path = '../../03.SavedModels/model_v7.pt'
test = get_predictions_fastercnn(df_test,config2,image_dir_test = '')

test.head()


# +
def get_string_pred(df,classes):   
    list_boxes = []
    list_labels = []
    list_string = []
    for idx in range(len(df)):
        if len(df.norm_bboxes.values[idx])==0:
            digits_labels  = []
            digits_boxes   = []
            string = '0-0-0'
        else:
            boxes  = df.post_boxes.values[idx]
            labels = df.post_labels[idx]
            date_bbox = _get_date(classes,boxes,labels)
            
            if date_bbox is None:
                    digits_labels = None
                    digits_boxes = None
                    string = '0-0-0'
            else:
                digits_labels,digits_boxes = _gate_digits(labels,boxes,date_bbox,classes)
                string = [classes[i] for i in digits_labels] 
                string = ''.join(string)
            
        list_boxes.append(digits_boxes)
        list_labels.append(digits_labels)
        list_string.append(string)
    df['fin_pred_boxes']  = list_boxes
    df['fin_pred_labels'] = list_labels
    df['date_txt'] = list_string
    return df

        
def _gate_digits(labels,boxes,date_bbox,classes):
    xmin = date_bbox[0]
    ymin = date_bbox[1]
    xmax = date_bbox[2]
    ymax = date_bbox[3]
    digits_boxes = []#[date_bbox]
    digits_labels = []#[12]
    digits_xc = []
    for i in range(len(boxes)):

        clase = classes[labels[i]]
        if clase !='date':
            xc = (boxes[i][0]+boxes[i][2])/2
            yc = (boxes[i][1]+boxes[i][3])/2
            if xc > xmin and xc < xmax and yc>ymin and yc<ymax:
                #pass
                digits_boxes.append(boxes[i])
                digits_labels.append(labels[i])
                digits_xc.append(xc)

    digits_boxes = [x for _,x in sorted(zip(digits_xc,digits_boxes))]
    digits_labels = [x for _,x in sorted(zip(digits_xc,digits_labels))]


    return digits_labels,digits_boxes

def _get_date(classes,boxes,labels):
    for i in range(len(boxes)):
        clase = classes[labels[i]]
        if clase =='date':
            return boxes[i]
    return None
def nms_one(boxes,scores,labels,nms):
    boxes = torch.tensor(boxes,dtype = torch.float32)
    scores = torch.tensor(scores,dtype = torch.float32)
    keep = torchvision.ops.nms(boxes, scores,nms)
    print(keep)
    boxes = boxes[keep].cpu().numpy().astype(np.float32)
    scores = scores[keep].cpu().numpy().astype(np.float32)
    labels = labels[keep]
    return boxes,scores,labels
    
def soft_nms_one(boxes,scores,labels,sigma,threshold):
    '''
    Soft nms - gaussian
    '''
    boxes = torch.tensor(boxes,dtype = torch.float32)
    scores = torch.tensor(scores,dtype = torch.float32)
    
    _,keep = snms(boxes,scores, sigma, threshold)
    boxes = boxes[keep].cpu().numpy().astype(np.float32)
    scores = scores[keep].cpu().numpy().astype(np.float32)
    labels = labels[keep]
    return boxes,scores,labels

def run_nms_after_preds(df,params,mode = 'nms'):
    '''
    - mode : ['nms','soft-nms']
    '''
    list_boxes = []
    list_scores = []
    list_labels = []
    for idx in range(len(df)):
        if len(df.norm_bboxes.values[idx])==0:
            boxes  = []
            labels = []
            scores = []
        else:
            boxes  = df.boxes.values[idx]
            labels = df.labels[idx]
            scores = df.scores[idx]
            if mode == 'nms':
                boxes, scores, labels = nms_one(boxes,scores,labels,params['iou_th'])
            elif mode == 'soft-nms':
                boxes, scores, labels = soft_nms_one(boxes,scores,labels,params['sigma'],params['threshold'])
        try:
            a = len(labels)
        except:
            labels = [labels]
        list_boxes.append(boxes)
        list_labels.append(labels)
        list_scores.append(scores)
        
    df['post_boxes']  = list_boxes
    df['post_labels'] = list_labels
    df['post_scores'] = list_scores
    return df
    

def denormalize_bboxes(boxes,size):
    # size = (25,170) # img.shape[1],img.shape[0] , #height,width
    denboxes = np.zeros(boxes.shape)
    denboxes[:,0] = boxes[:,0]*size[1]
    denboxes[:,1] = boxes[:,1]*size[0]
    denboxes[:,2] = boxes[:,2]*size[1]
    denboxes[:,3] = boxes[:,3]*size[0]
    return denboxes

def normalize_bboxes(boxes,size):
    # size = (25,170) # img.shape[1],img.shape[0] , #height,width
    nboxes = np.zeros(boxes.shape)
    nboxes[:,0] = boxes[:,0]/size[1]
    nboxes[:,1] = boxes[:,1]/size[0]
    nboxes[:,2] = boxes[:,2]/size[1]
    nboxes[:,3] = boxes[:,3]/size[0]
    return nboxes
params = {'iou_th':0.14}
test = run_nms_after_preds(test.copy(),mode = 'nms',params = params)
test = get_string_pred(test,data_dict)
# -

test[['id','date_txt']]


def apply_encoding(row):
    if row['date_txt']=='0-0-0':
        return '0','0','0'
    #####
    if row['date_txt'].count('-')==3:
        return row['date_txt'].split('-')[0],row['date_txt'].split('-')[0],'2021'
    ####
    val = row['date_txt'].split('-')
    val = [i for i in val if (i!=' ')&(i!='')]
    vector = [np.nan,np.nan,np.nan]
    for i in range(len(vector)):
        try:
            vector[i] = val[i]
        except:
            pass
    try:
        if len(vector[2])>2:
            vector[2] = '2021'
        elif len(vector[2])==2:
            vector[2] = '21'
        else:
            vector[2] = '2021'
    except:
        pass
    
    try:
        if len(vector[1])>2:
            vector[1] = '05'
    except:
        pass
    
    try:
        if len(vector[0])>2:
            vector[1] = '04'
    except:
        pass
    return vector[0],vector[1],vector[2]


test[['date_day','date_month','date_year']] = test.apply(lambda x:apply_encoding(x),axis = 1,result_type='expand')
test['date_day'] = test['date_day'].fillna('04')
test['date_month'] = test['date_month'].fillna('05')
test['date_year'] = test['date_year'].fillna('2021')

df_submission = pd.read_csv('submission_fin.csv')



df_submission[['id','sign_1','sign_2']].merge(test[['id','date_day','date_month','date_year']], on = ['id']).to_csv('prueba.csv',index = False)


