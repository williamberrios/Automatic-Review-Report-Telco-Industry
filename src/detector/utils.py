# +
import cv2
import os
from tqdm import tqdm

def processing_images(df,my_detector,DATA_PATH,mode = 'images_train'):
    '''
    Function that performs: 1.Alignment 2.Extract firms and dates
    
    Parameters:
    - df        : Dataframe with information needed
    - DATA_PATH : General Path with datasets
    - mode      : images for preprocessing: [images_train,images_test]
    
    '''
    for idx in tqdm(range(len(df))):
        id_img  = df.loc[idx,'id']
        img     = cv2.imread(os.path.join(DATA_PATH,mode,f'{id_img}.jpg'), cv2.IMREAD_COLOR)
        img_aligned, fecha, firma1, firma2 = my_detector.predict(img)
        # Saving img_aligned:
        filename = os.path.join(DATA_PATH,'processed',mode,'aligned',f'{id_img}.jpg')
        cv2.imwrite(filename,img_aligned)
        # Saving fecha:
        filename = os.path.join(DATA_PATH,'processed',mode,'fechas',f'{id_img}.jpg')
        cv2.imwrite(filename,fecha)
        # Saving firma1:
        #filename = os.path.join(DATA_PATH,'processed',mode,'firma1',f'{id_img}.jpg')
        #cv2.imwrite(filename,firma1)
        filename = os.path.join(DATA_PATH,'processed',mode,'firmas',f'{id_img}_1.jpg')
        cv2.imwrite(filename,firma1)
        
        # Saving firma2:
        #filename = os.path.join(DATA_PATH,'processed',mode,'firma2',f'{id_img}.jpg')
        #cv2.imwrite(filename,firma2)
        filename = os.path.join(DATA_PATH,'processed',mode,'firmas',f'{id_img}_2.jpg')
        cv2.imwrite(filename,firma2)
