# +
# General Parameters
seed           = 42
epochs         = 250
device         = 'cuda'
lr             = 1e-4
batch_size     = 8
num_workers    = 72 
image_dir      = ''
output_path    = '../../03.SavedModels/model_fechas.pt'

# Model Parameters
name_model            = 'resnet152'
num_classes           = 13 
pretrained_coco       = False
pretrained_back       = True

# Early stopping:
early_stopping = 5
mode           = 'min'
# Optimizer:
optimizer_params = {'name':'Adam',
                         'WD'  : 0}
# Scheduler:
scheduler_params = {'name':'Plateu',
                         'mode':None,
                         'step_metric':'train_loss',
                         'patience':2}
