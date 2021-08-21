import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import numpy as np

from tensorflow import keras
from .utils import custom_f1



def evaluate_data(model,path_dataset,tipo,WIDTH,HEIGHT):
    print('Analizando ',tipo)
    path = os.path.join(path_dataset,tipo)
    predictions = []
    y_test = []
    filename_list = glob.glob(os.path.join(path,'0','*.jpg'))
    for filename in tqdm(filename_list):
        img_rgb1 = plt.imread(filename)
        img_rgb = cv2.resize(img_rgb1,(WIDTH,HEIGHT))
        img_rgb = img_rgb/255.0
        img = np.reshape(img_rgb,(-1,HEIGHT, WIDTH,3))
        prediction = model.predict(img)
        clase = 0 if prediction[0][0]< 0.49 else 1
        predictions.append(clase)
        y_test.append(0)

    filename_list = glob.glob(os.path.join(path,'1','*.jpg'))
    for filename in tqdm(filename_list):
        img_rgb1 = plt.imread(filename)
        img_rgb = cv2.resize(img_rgb1,(WIDTH,HEIGHT))
        img_rgb = img_rgb/255.0
        img = np.reshape(img_rgb,(-1,HEIGHT, WIDTH,3))
        prediction = model.predict(img)
        clase = 0 if prediction[0][0]< 0.49 else 1
        predictions.append(clase)
        y_test.append(1)
        
        
    roc = roc_auc_score(predictions,y_test)
    f1  = f1_score(y_test, predictions, average='micro')
    print('roc',roc)
    print('f1',f1)

    labels = [0, 1]
    cm = confusion_matrix(y_test, predictions, labels)
    fig_cm = plt.figure(figsize=(5,3))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(tipo+' Confusion matrix - roc '+str(np.round(roc,2))+" f1 "+str(np.round(f1,2))); 
    ax.xaxis.set_ticklabels([0, 1]); ax.yaxis.set_ticklabels([0, 1]);
    

    fig_cm.savefig("graficos/confusion_matrix_"+tipo+".png", dpi=fig_cm.dpi)
    print('Resultados de ',tipo)
    print('-----------------------')
    print(" roc :"+str(np.round(roc,2)))
    print(" f1  :"+str(np.round(f1,2)))

    return predictions,y_test,fig_cm,roc,f1

def evaluate_model(PATH_MODEL,PATH_DATASET,WIDTH,HEIGHT):

    model = keras.models.load_model(PATH_MODEL, custom_objects={"custom_f1": custom_f1},compile=False)

    
    predictions,y_test,fig_cm,roc,f1 = evaluate_data(model,PATH_DATASET,'train',WIDTH,HEIGHT)

    predictions,y_test,fig_cm,roc,f1 = evaluate_data(model,PATH_DATASET,'validation',WIDTH,HEIGHT)

