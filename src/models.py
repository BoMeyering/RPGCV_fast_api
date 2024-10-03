# Effdet creation
# BoMeyering 2024

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

import segmentation_models_pytorch as smp

def create_effdet_model(num_classes=2, image_size=1024, architecture="tf_efficientdet_d4"):
   """
   Instantiate an effdet model
   """
   config = get_efficientdet_config(architecture)
   config.update({'num_classes': num_classes})
   config.update({'image_size': (image_size, image_size)})

   net = EfficientDet(config, pretrained_backbone=False)
   net.class_net = HeadNet(
      config,
      num_outputs=config.num_classes
   )

   return DetBenchPredict(net)

def create_deeplabv3plus_model(num_classes=1, in_channels=1, encoder_name="efficientnet-b0", encoder_depth=2, encoder_weights="imagenet"):
    """
    Instantiate a DeepLabV3Plus model
    """
    model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels, 
            classes=num_classes
        )
    
    return model
    
      
      
      
