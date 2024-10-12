# Effdet creation
# BoMeyering 2024

import torch
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

import segmentation_models_pytorch as smp

def load_state_dict(device: str,):
   state_dict = torch.load(
      "model_chkpt/marker_effdet-epoch=19-val_loss=0.113.ckpt", 
      map_location=torch.device(device)
   )['state_dict']
    
   new_state_dict = {}
   for k, v in state_dict.items():
      if k == 'model.anchors.boxes':
         k = 'anchors.boxes'
      k = k.replace('model.model', 'model')
      new_state_dict[k] = v
    
   return new_state_dict

def create_effdet_model(device):
   """
   Instantiate an effdet model in inference mode
   """
   config = get_efficientdet_config("tf_efficientdet_d5")
   config.update({'num_classes': 2})
   config.update({'image_size': (1024, 1024)})

   net = EfficientDet(config, pretrained_backbone=False)
   net.class_net = HeadNet(
      config,
      num_outputs=config.num_classes
   )

   state_dict = load_state_dict(device)
    
   model = DetBenchPredict(net)
   model.load_state_dict(state_dict)
   model = model.to(device)
   model.eval()
   
   return model

def create_deeplabv3plus_model(device):
   """
   Instantiate a DeepLabV3Plus model in inference mode
   """
   model = smp.DeepLabV3Plus(
      encoder_name='efficientnet-b4', 
      encoder_depth=5, 
      encoder_weights='imagenet', 
      in_channels=3, 
      classes=8
   )

   state_dict = torch.load(
      'model_chkpt/fixmatch_dlv3p_1024_enb4_regenpgc_set_2024-07-12_13.23.20_epoch_18_2024-07-13_04.01.58', 
      map_location=device
   )['model_state_dict']
    
   model.load_state_dict(state_dict)
   model = model.to(device)
   model.eval()

   return model
    
      
      
      
