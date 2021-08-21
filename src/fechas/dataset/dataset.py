# +
import torch
import numpy as np
import pandas as pd
import cv2
import os


class DateDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir,transforms=None,mode = 'train'):
        self.image_ids  = dataframe['filename'].unique()
        self.df         = dataframe
        self.image_dir  = image_dir
        self.transforms = transforms
        self.mode       = mode

    def __getitem__(self, idx):
        # Get Image ID
        image_id = self.image_ids[idx]
        # Get All records
        records  = self.df[self.df['filename'] == image_id]
        # Get Image
        image = cv2.imread(os.path.join(self.image_dir,records['filename_path'].unique()[0]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.mode == 'train':
            # Get Boxes
            boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
            # Calculate Area
            area = self._CalArea(boxes)
            # Get Labels
            labels = torch.as_tensor(records['label'].values, dtype=torch.int64)
            # suppose all instances are not crowd (Not too much overlaping - True for this case)
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

            target = {}
            target['boxes']    = boxes
            target['labels']   = labels
            target['image_id_name'] = image_id
            target['image_id'] = torch.tensor([idx])
            target['area'] = area
            target['iscrowd'] = iscrowd

            if self.transforms:
                sample = {'image': image,'bboxes': target['boxes'],'labels': labels}
                image = self.transforms(**sample)
                image = image['image']
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            # COnvert coordinates to float32
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32) 
            return image, target
            
        elif self.mode == 'test':
            if self.transforms:
                sample = {
                    'image': image,
                }
                sample = self.transforms(**sample)
                image = sample['image']

            return image, image_id

    def __len__(self):
        return self.image_ids.shape[0]
    
    def _CalArea(self,boxes):
        area =  (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1])  
        return torch.as_tensor(area, dtype=torch.float32)
