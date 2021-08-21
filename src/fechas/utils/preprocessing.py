# +
import glob
import os
import cv2
import pandas as pd

def generate_df(DATA_PATH,dict_classes,mode = 'train'):
    images_list = glob.glob(os.path.join(DATA_PATH,'labeling',mode,'*.txt'))[:-1]
    data = []
    for filename in images_list:
        image_name = f"{filename.split('/')[-1].split('.')[0]}"
        # Opening filename with labels
        file       = open(filename,'r').readlines()
        lines      = [i.replace('\n','') for i in file]
        img_path   = os.path.join(DATA_PATH,'processed',f'images_{mode}','fechas',f"{image_name}.jpg")
        img        = cv2.imread(img_path)
        for line in lines:
            #==================================================================
            #================= Convert to Xmin, Xmax, Ymin, Ymax ==============
            #==================================================================
            # Yolo Format: <object-class> <x> <y> <width> <height>
            # <x> , <y>: center point of the bounding box
            #<x> = <absolute_x> / <image_width> 
            #<y> = <absolute_y> / <image_height> 
            #<height> = <absolute_height> / <image_height>
            #<width> = <absolute_width> / <image_width>
            info = line.split(' ')
            label,x, y, width, height = int(info[0]),float(info[1]),float(info[2]),float(info[3]),float(info[4])
            xmin   = int((x - width/2 )*img.shape[1])
            ymin   = int((y - height/2)*img.shape[0])
            xmax   = int((x + width/2 )*img.shape[1])
            ymax   = int((y + height/2)*img.shape[0])
            width  = xmax-xmin
            height = ymax-ymin 
            label_txt = dict_classes[label + 1]
            data.append({'filename':image_name,'filename_path':os.path.join(DATA_PATH,'processed',f'images_{mode}','fechas',f"{image_name}.jpg"),'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'w':width,'h':height,'label_txt':label_txt,'label':label + 1})
    return pd.DataFrame(data)


