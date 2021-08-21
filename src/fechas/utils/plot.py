# +
import matplotlib.pyplot as plt
import cv2
import numpy as np
def plot_date(image,target,MEAN,STD,mode_transforms = False):
    if mode_transforms:
        image = (image.permute(1,2,0)*np.array(STD) + np.array(MEAN))*255
        image = image.numpy()
    image = image.astype(np.uint8)
    for i in target['boxes']:
        points = i.numpy()
        x_min,y_min,x_max,y_max = int(points[0]),int(points[1]),int(points[2]),int(points[3])
        start_point = (x_min,y_min)
        end_point   = (x_max,y_max)
        color      = (255,0,0) 
        thickness = 1
        image = cv2.rectangle(image,start_point, end_point, color, thickness)
        plt.imshow(image)
        
def plot_boxes_predicted(df,idxs = [0],data_dict = {}):
    for idx in idxs:
        row = df.iloc[idx,:]
        boxes = row.post_boxes
        labels = row.post_labels
        image = cv2.imread(row.filename_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        print('Image: ',row.id,', Date Text: ',row.date_target)
        plot_predict_one(image,boxes,labels,data_dict)

def plot_predict_one(image,boxes,labels,data_dict):
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.show()
    if boxes is None:
        pass
    else:
        
        for i in range(len(boxes)):
            points = boxes[i]
            x_min,y_min,x_max,y_max = int(points[0]),int(points[1]),int(points[2]),int(points[3])
            start_point = (x_min,y_min)
            end_point   = (x_max,y_max)
            color      = (255,0,0) 
            thickness = 1

            image = cv2.rectangle(image,start_point, end_point, color, thickness)
            image = cv2.putText(image, data_dict[labels[i]], (points[0],points[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(220,0,0),1,cv2.LINE_AA)
    plt.imshow(image)
    plt.show()
