from tensorflow import keras
import tensorflow.keras.backend as K
from numpy.linalg import norm
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class firmas_model:
    def __init__(self,filename):
        
        self.model = keras.models.load_model(filename, custom_objects={"custom_f1": self.custom_f1},compile=False)
        self.WIDTH = 350
        self.HEIGHT = 100

    
    def custom_f1(self,y_true, y_pred):    
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
    
    def preprocessing(self,img_rgb):
        
        cv2.imwrite('auxiliar.jpg',img_rgb)
        img_rgb = plt.imread('auxiliar.jpg')
        os.remove('auxiliar.jpg')
        img_rgb = cv2.resize(img_rgb,(self.WIDTH,self.HEIGHT))
        img_rgb = img_rgb/255.0
        img = np.reshape(img_rgb,(-1,self.HEIGHT, self.WIDTH,3))
        return img
    def predict(self,img_rgb,threshold = 0.49):
        img = self.preprocessing(img_rgb)
        prediction = self.model.predict(img)
        clase = 0 if prediction[0][0]< threshold else 1
        return clase,prediction[0][0]