# +
import torch
import torchvision
import numpy as np
import pandas as pd
import sys
module_path = "../../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from fechas.dataset.dataset import DateDataset
from fechas.utils.seed import seed_everything
from fechas.utils.AverageMeter import AverageMeter
from fechas.utils.EarlyStopping import EarlyStopping
import os
import time
from tqdm import tqdm

class TrainerFasterRCNN:
    def __init__(self,
                 config = None,
                 model = None,
                 dict_transforms = {'train':None,'valid':None,'test':None}
                ):
        self.config    = config
        self.dict_transforms = dict_transforms
        # Calculating the Model, Scheduler, Optimizer and Loss
        self.model     = model.to(self.config.device)
        #self.criterion = self._fetch_loss()
        self.optimizer = self._fetch_optimizer()
        self.scheduler = self._fetch_scheduler()
    
    def train_fn(self,train_loader):
        # Model: train-mode
        self.model.train()
        outputs = []
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(train_loader,total = len(train_loader))
        # Reading batches of data
        for b_idx,data in enumerate(tk0):
            images,targets = data
            images    = images.to(self.config.device)
            targets   = [{k: v.to(self.config.device) for k, v in t.items() if k!= 'image_id_name'} for t in targets]
            # Zero grading optimizer
            self.optimizer.zero_grad()
            # Calculating the loss
            loss_dict = self.model(images,targets) 
            loss    = sum(loss for loss in loss_dict.values())        
            # Calculate gradients
            loss.backward()
            # Update Optimizer
            self.optimizer.step()
            # Update Scheduler
            if (self.config.scheduler_params['mode'] == 'batch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
                self.scheduler.step()
            losses.update(loss.detach().item(), train_loader.batch_size)
            tk0.set_postfix(Train_Loss = losses.avg, LR = self.optimizer.param_groups[0]['lr'])
        if (self.config.scheduler_params['mode']=='epoch')&(self.config.scheduler_params['name'] not in ['Plateu',None]):
                self.scheduler.step()
        return losses.avg
    
    def valid_fn(self,valid_loader):
        self.model.eval()
        validation_image_precisions = []
        iou_thresholds = [x for x in np.arange(0.5)]
        # Initialize object Average Meter
        losses = AverageMeter()
        tk0 = tqdm(valid_loader,total = len(valid_loader))
        for b_idx,data in enumerate(tk0):
            images,targets = data
            images    = images.to(self.config.device)
            targets   = [{k: v.to(self.config.device) for k, v in t.items() if k!= 'image_id_name'} for t in targets]
            
            with torch.no_grad():
                outputs = self.model(images)
            
            
            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                preds_sorted_idx = np.argsort(scores)[::-1]
                preds_sorted = boxes[preds_sorted_idx]
                image_precision = calculate_image_precision(preds_sorted,
                                                            gt_boxes,
                                                            thresholds=iou_thresholds,
                                                            form='pascal')
                validation_image_precisions.append(image_precision)
        valid_prec = np.mean(validation_image_precisions)
        return valid_prec
              
    
    def fit(self,train = None,valid = None,advance = False,output_path = None):
        # seed everything for reproducibilty
        seed_everything(self.config.seed)
        # Creating datasets for training and Validation

        train_dataset = DateDataset(train,self.config.image_dir,self.dict_transforms['train'])
        valid_dataset = DateDataset(valid,self.config.image_dir,self.dict_transforms['valid'])
        
        # Creating Dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   pin_memory  = True,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = True,
                                                   collate_fn=collate_fn)
        
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size  = self.config.batch_size,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   collate_fn  = collate_fn)

        self.model.to(self.config.device)
        es = EarlyStopping (patience = self.config.early_stopping, mode = self.config.mode, delta = 0)
        
        for epoch in range(self.config.epochs):
            print(f'=========== EPOCH: {epoch + 1} ===========')
            time.sleep(1)
            train_loss = self.train_fn(train_loader)
            valid_loss = self.valid_fn(valid_loader)
            print(f"Validation Loss: {np.round(train_loss,4)} & Validation IOU [0.5]: {np.round(valid_loss,4)}")
            valid_metrics = {'valid_loss':valid_loss,'train_loss':train_loss} # Could be whaetver metric to follow
            if self.config.scheduler_params['step_metric'] != None: 
                self.scheduler.step(valid_metrics[self.config.scheduler_params['step_metric']])    
            es(valid_metrics[self.config.scheduler_params['step_metric']], self.model,output_path)
            if es.early_stop:
                print('Meet early stopping')
                self._clean_cache()
                return es.get_best_val_score()
            
        self._clean_cache()
        print("Didn't meet early stopping")
        return es.get_best_val_score()
        
    def predict(self,test = None,image_dir_test = './'):
        self.model.eval()
        test_dataset = DateDataset(test,image_dir_test,self.dict_transforms['test'],mode = 'test')
        test_loader  = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size  = 1,
                                                   pin_memory  = True,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = False)
        # Get Prediction:
        list_boxes  = []
        list_labels = []
        list_img_name = []
        list_scores   = []
        list_norm_bboxes = []
        list_size        = []
        for b_idx,data in tqdm(enumerate(test_loader)):
            images,img_name = data
            img_name  = img_name[0]
            images    = images.to(self.config.device)
            size = (images.size()[2],images.size()[3])
            
            with torch.no_grad():
                outputs = self.model(images)
                outputs = [{k: v.to(self.config.device) for k, v in t.items()} for t in outputs]
                boxes = outputs[0]['boxes'].data.cpu().numpy()
                scores = outputs[0]['scores'].data.cpu().numpy()
                labels = outputs[0]['labels'].data.cpu().numpy()
                nboxes = self._normalize_bboxes(boxes,size)
                list_img_name.append(img_name)
                list_boxes.append(boxes)
                list_labels.append(labels)
                list_scores.append(scores)
                list_norm_bboxes.append(nboxes)
                list_size.append(size)
        return pd.DataFrame({'id':list_img_name,'boxes':list_boxes,'labels':list_labels,'scores':list_scores,'norm_bboxes':list_norm_bboxes,'img_size':list_size})
        
        
    def predict_nms(self,test = None,image_dir_test = None,classes = None,nms = 0.2):
        self.model.eval()
        test_dataset = DateDataset(test,image_dir_test,self.dict_transforms['test'],mode = 'test')
        test_loader  = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size  = 1,
                                                   pin_memory  = True,
                                                   num_workers = self.config.num_workers,
                                                   shuffle     = False)
        # Get Prediction:
        list_boxes  = []
        list_labels = []
        list_string = []
        list_img_name = []

        for b_idx,data in tqdm(enumerate(test_loader)):
            images,img_name = data
            img_name = img_name[0]
            images    = images.to(self.config.device)
            with torch.no_grad():
                outputs = self.model(images)
                outputs = [{k: v.to(self.config.device) for k, v in t.items()} for t in outputs]
                boxes = outputs[0]['boxes']
                scores = outputs[0]['scores']
                labels = outputs[0]['labels']
                
                keep = torchvision.ops.nms(boxes, scores,nms)
                boxes = boxes[keep].cpu().numpy().astype(np.int32)
                scores = scores[keep].cpu().numpy().astype(np.int32)
                labels = labels[keep].data.cpu().numpy()

                date_bbox = self._get_date(classes,boxes,labels)
                if date_bbox is None:
                    digits_labels = None
                    digits_boxes = None
                    string = '0-0-0'
                else:
                    digits_labels,digits_boxes = self._gate_digits(labels,boxes,date_bbox,classes)
                    string = [classes[i] for i in digits_labels] 
                    string = ''.join(string)
                list_img_name.append(img_name)
                list_boxes.append(digits_boxes)
                list_labels.append(digits_labels)
                list_string.append(string)
        
        return pd.DataFrame({'id':list_img_name,'boxes':list_boxes,'labels':list_labels,'date_txt':list_string})
        
    def _gate_digits(self,labels,boxes,date_bbox,classes):
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
    
    def _get_date(self,classes,boxes,labels):
        for i in range(len(boxes)):
            clase = classes[labels[i]]
            if clase =='date':
                return boxes[i]
        return None
        
    def _normalize_bboxes(self,boxes,size):
        # size = (25,170) # img.shape[1],img.shape[0] , #height,width
        nboxes = np.zeros(boxes.shape)
        nboxes[:,0] = boxes[:,0]/size[1]
        nboxes[:,1] = boxes[:,1]/size[0]
        nboxes[:,2] = boxes[:,2]/size[1]
        nboxes[:,3] = boxes[:,3]/size[0]
        return nboxes
    def _fetch_scheduler(self):
        '''
        Add any scheduler you want
        '''    
        if self.optimizer is None:
            raise Exception('First choose an optimizer')
        
        else:
            sch_params = self.config.scheduler_params
            
            if sch_params['name'] == 'StepLR':
                return torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                       step_size = sch_parmas['step_size'], 
                                                       gamma     = sch_params.get('gamma',0.1)
            elif sch_params['name'] == 'Plateu': 
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  mode      = self.config.mode, 
                                                                  factor    = sch_params.get('factor',0.1), 
                                                                  patience  = sch_params['patience'], 
                                                                  threshold = 0)
            elif sch_params['name'] == None:
                return None
            else:
                raise Exception('Please choose a valid scheduler')                                       
                
        
    def _fetch_optimizer(self):
        '''
        Add any optimizer you want
        '''
        op_params = self.config.optimizer_params
                                                       
        if op_params['name'] == 'Adam':
            return torch.optim.Adam(self.model.parameters(),lr = self.config.lr, weight_decay = op_params.get('WD',0))
        if op_params['name'] == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr = self.config.lr , weight_decay =  op_params.get('WD',0))
        else: 
            raise Exception('Please choose a valid optimizer')
    
    def _clean_cache(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# -
def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets=list()
    for i, t in batch:
        images.append(i)
        targets.append(t)
    images = torch.stack(images, dim=0)

    return images, targets


def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1
    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1
        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision
