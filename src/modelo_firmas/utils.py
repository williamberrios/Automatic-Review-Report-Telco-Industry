import tensorflow.keras.backend as K
import os
import random
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt


# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# Define our metric
def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Seed
def seed_everything(seed=42):
    '''
    
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

    
def plot_history(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    score_f1= history.history['custom_f1']
    val_score_f1= history.history['val_custom_f1']

    score_auc= history.history['auc']
    val_score_auc= history.history['val_auc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(history.epoch) + 1)


    fig_model_performance = plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(epochs_range, acc, label='Training')
    plt.plot(epochs_range, val_acc, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Model accuracy')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()

    plt.subplot(2,2,2)

    plt.plot(epochs_range, loss, label='Training')
    plt.plot(epochs_range, val_loss, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()

    plt.subplot(2,2,3)
    plt.plot(epochs_range, score_f1, label='Training')
    plt.plot(epochs_range, val_score_f1, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('score_f1')
    plt.title('Model score_f1')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()

    plt.subplot(2,2,4)
    plt.plot(epochs_range, score_auc, label='Training')
    plt.plot(epochs_range, val_score_auc, label='Validation')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('score_auc')
    plt.title('Model score_auc')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.tight_layout()
    plt.show()

    return fig_model_performance