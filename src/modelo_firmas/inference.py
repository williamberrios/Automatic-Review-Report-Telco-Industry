from tensorflow import keras
import tensorflow.keras.backend as K
from numpy.linalg import norm
import cv2
import numpy as np

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
    
    

    def brightness(self,img):
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)
    def adjust_gamma(self,image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)    
    
    def predict(self,img_rgb,threshold = 0.49):
        #if self.brightness(img_rgb)<101:
        #    img_rgb = self.adjust_gamma(img_rgb,1.25)/255.0
        #else:
        #    img_rgb = img_rgb.copy()/255.0
        img_rgb = cv2.resize(img_rgb,(self.WIDTH,self.HEIGHT))
        img_rgb = img_rgb/255.0
        img = np.reshape(img_rgb,(-1,self.HEIGHT, self.WIDTH,3))
        #img = cv2.resize(img_rgb,(150,150))
        #img = np.reshape(img,(-1,150,150,3))
        prediction = self.model.predict(img)
        clase = 0 if prediction[0][0]< threshold else 1
        return clase,prediction[0][0]