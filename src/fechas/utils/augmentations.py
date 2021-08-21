# +
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(name = 'train',MEAN = None,STD = None):
        if name == 'train':
            return  A.Compose([
                    A.Normalize(
                    mean = MEAN,
                    std = STD,
                    ),
                    A.geometric.rotate.Rotate (limit=3,p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.5),        
                    ToTensorV2()],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels']))
            
        elif name == 'valid':
            return A.Compose([
                    A.Normalize(
                    mean = MEAN,
                    std = STD,
                    ),
                    ToTensorV2()],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels']))
        elif name == 'test':
            return A.Compose([
                    A.Normalize(
                    mean = MEAN,
                    std = STD,
                    ),
                    ToTensorV2()])


def get_transformations(MEAN,STD):
        return {'train' : get_transform(name = 'train',MEAN = MEAN,STD = STD),
                'valid' : get_transform(name = 'valid',MEAN = MEAN,STD = STD),
                'test' : get_transform(name = 'test',MEAN = MEAN,STD = STD)}
