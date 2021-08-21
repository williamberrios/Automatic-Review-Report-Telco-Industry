# +
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FasterRCNNTorchVision(nn.Module):
    def __init__(self,name_model,n_classes,pretrained_coco = False,pretrained_back = False,trainable_layers = 3):
        super(FasterRCNNTorchVision, self).__init__()
        self.name_model = name_model
        self.pretrained_back = pretrained_back
        self.pretrained_coco = pretrained_coco
        self.n_classes = n_classes
        self.trainable_layers = trainable_layers
        self.base_model = self.GetModel()
        # Number of input featuresFasterRCNNTorchVision
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_classes)
        

    def GetModel(self):
        if self.name_model == 'resnet_50':
            return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained_coco,pretrained_backbone = self.pretrained_back)
        elif 'res' in self.name_model:
            print(f"{self.name_model} does not have coco weights ")
            return self._faster_rcnn_resnet_models(name_model =self.name_model,pretrained_backbone=self.pretrained_back,num_classes=self.n_classes,trainable_layers = self.trainable_layers)

        else:
            print('Model Not implemented')

    def ReturnModel(self):
        return self.base_model
    
    def _faster_rcnn_resnet_models(self,name_model = 'resnet101',
                                   pretrained_backbone=True,
                                   num_classes=13,
                                   trainable_layers = 3):
        #  backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
        #  'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        backbone = resnet_fpn_backbone(name_model, pretrained_backbone,trainable_layers = trainable_layers)
        model = FasterRCNN(backbone, num_classes)
        #in_features = model.roi_heads.box_predictor.cls_score.in_features
        #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
