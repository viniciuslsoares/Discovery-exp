from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone
from minerva.models.loaders import FromPretrained
from torchvision.models.resnet import resnet50
from minerva.models.nets.image.deeplabv3 import DeepLabV3

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torchvision.models
import torch
import torch.nn as nn

from pathlib import Path
import re

import torch
import torch.nn as nn
from auxiliar import *
from data import *
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.linear_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_head(x)


def get_state_dict(model):
    state_dict = model.state_dict()
    renamed_state_dict = {}
    for key in state_dict.keys():
        # Replace the key prefix to match my_state_dict
        new_key = f"RN50model.{key}" if not key.startswith("RN50model.") else key
        renamed_state_dict[new_key] = state_dict[key]

    return renamed_state_dict    


# def get_seg_file(root_path="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ckpt/train", 
#                  repetition=0,
#                  seg_data="seam_ai_N",):
    
#     folder_path = f"{root_path}/{repetition}/V{repetition}_pre_sup_train_{seg_data}_cap_100%/{seg_data}"
    
#     files = [f for f in Path(folder_path).iterdir() if f.is_file() and f.name.startswith("epoch=")]
#     if files:
#         files.sort(key=lambda f: extract_epoch_number(f.name))
#         selected_file = files[0]
#         return str(selected_file.resolve())

#     return None

def get_seg_file(root_path, repetition, seg_data):
    """
    Busca o checkpoint de um modelo treinado de forma supervisionada (pre_sup)
    para ser usado como inicialização em outro dataset.
    """

    folder_name = f"V{repetition}_pre_sup_train_{seg_data}_cap_100%"
    target_dir = Path(root_path) / str(repetition) / folder_name / seg_data
    
    if not target_dir.exists():
        logger.error(f"Diretório de pesos supervisionados não encontrado: {target_dir}")
        return None
    

    files = [f for f in target_dir.iterdir() if f.is_file() and f.name.endswith(".ckpt")]
    
    if files:
        last_ckpt = [f for f in files if "last" in f.name]
        if last_ckpt:
            return str(last_ckpt[0].resolve())
            
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return str(files[0].resolve())

    return None



def print_trainable_params(model: torch.nn.Module):
    total_params = 0
    print("\n=== Trainable parameters by layer ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:50} {str(tuple(param.shape)):20} -> {num_params:,}")
    print("------------------------------------------------------------")
    print(f"Total trainable parameters: {total_params:,}")


def apply_freeze(backbone: DeepLabV3Backbone, freeze_list: list):
    """
    Aplica freeze seletivo nos blocos do DeepLabV3Backbone.
    freeze_list = [conv1+bn1+relu+maxpool, layer1, layer2, layer3, layer4]
    """
    rn50 = backbone.RN50model

    # Freeze stem (conv1 + bn1 + relu + maxpool)
    if freeze_list[0]:
        for param in rn50.conv1.parameters():
            param.requires_grad = False
        for param in rn50.bn1.parameters():
            param.requires_grad = False
        for param in rn50.fc.parameters():
            param.requires_grad = False

    # Freeze ResNet layers
    for idx, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
        if freeze_list[idx]:
            layer = getattr(rn50, layer_name)
            for param in layer.parameters():
                param.requires_grad = False
                
    if freeze_list[4]:
        for param in rn50.avgpool.parameters():
            param.requires_grad = False
                
    # Print the backbone summary
    # print_trainable_params(rn50)
    # print("******************")
    # trainable = sum(p.numel() for p in rn50.parameters() if p.requires_grad)

    return backbone


def apply_layerwise_lr(backbone: DeepLabV3Backbone, lr_config: dict):
    """
    Applies different learning rates (or freezing) to specific ResNet layers
    inside a DeepLabV3Backbone.

    Args:
        backbone (DeepLabV3Backbone): The DeepLabV3 backbone with RN50model.
        lr_config (dict): A mapping of layer names to learning rates.
            Example:
                {
                    "stem": 0.0,       # freeze conv1/bn1/fc
                    "layer1": 1e-5,
                    "layer2": 1e-4,
                    "layer3": 1e-4,
                    "layer4": 1e-3,
                    "avgpool": 0.0     # optional
                }

    Returns:
        (backbone, param_groups): the backbone and a list of parameter groups
        for optimizer creation.
    """
    rn50 = backbone.RN50model
    param_groups = []

    def add_group(params, lr):
        params = [p for p in params if p.requires_grad]
        if len(params) > 0:
            param_groups.append({"params": params, "lr": lr})

    # Stem
    if "stem" in lr_config:
        lr = lr_config["stem"]
        for param in list(rn50.conv1.parameters()) + list(rn50.bn1.parameters()) + list(rn50.fc.parameters()):
            param.requires_grad = lr > 0
        add_group(list(rn50.conv1.parameters()) + list(rn50.bn1.parameters()) + list(rn50.fc.parameters()), lr)

    # ResNet layers
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        if layer_name in lr_config:
            lr = lr_config[layer_name]
            layer = getattr(rn50, layer_name)
            for param in layer.parameters():
                param.requires_grad = lr > 0
            add_group(layer.parameters(), lr)

    # Avgpool
    if "avgpool" in lr_config:
        lr = lr_config["avgpool"]
        for param in rn50.avgpool.parameters():
            param.requires_grad = lr > 0
        add_group(rn50.avgpool.parameters(), lr)

    return backbone, param_groups


# def get_model(
#     pretrain_data,
#     learning_rate,
#     freeze,
#     repetition,
#     root_path=None,
#     full_path=False,
#     linear=False,
#     finetune_data=None,
#     ):
#     num_classes = 6

#     if full_path: 
#         import_path = full_path
#     else:
#         if pretrain_data == "a700":
#             base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B128_E125000_lr1e-05"
#         elif pretrain_data == "namss":
#             base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B128_E125000_lr1e-05"
#         else:
#             base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E1200_lr1e-05"

#         if root_path:
#             import_path = f"{root_path}/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
#         else:
#             import_path = (
#                 f"ckpt/pretrain/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
#             )
            
#     if pretrain_data == "seg":
#         seg_data_mapping = {
#             "f3_N": "seam_ai_N",
#             "f3": "seam_ai",
#             "seam_ai_N": "f3_N",
#             "seam_ai": "f3",
#         }
#         root_path = "/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/checkpoints/ckpt_vinicius/train"
#         if finetune_data in ["f3_N", "seam_ai_N", "f3", "seam_ai"]:
#             import_path = get_seg_file(
#                 root_path=root_path,
#                 repetition=repetition,
#                 seg_data=seg_data_mapping[finetune_data],
#             )
#             logger.info(f"CKPT imported from:\n{import_path}")
#             logger.info(f"Imported pretrain for segmentation in {seg_data_mapping[finetune_data]}")
#         else:
#             raise ValueError("Invalid finetune_data for pretrain_data 'seg'")

#     seg_data = ["f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700", "namss"]

#     resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)

#     if pretrain_data in seg_data:
#         # Builds the BYOL model, loads the weights and extracts the backbone
#         model = BYOL(backbone=resnet50_backbone, learning_rate=learning_rate)
#         logger.info(import_path)
#         resnet50_backbone = FromPretrained(
#             model=model,
#             ckpt_path=import_path,
#             strict=False,
#             error_on_missing_keys=False,
#         ).backbone
#         logger.info(f"{pretrain_data} backbone loaded")

#     elif pretrain_data == "imagenet":
#         weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
#         imagenet_backbone = resnet50(
#             replace_stride_with_dilation=[False, True, True], weights=weights
#         )
#         imagenet_state_dict = get_state_dict(imagenet_backbone)
#         resnet50_backbone.load_state_dict(imagenet_state_dict, strict=False)
#         logger.info("IMAGENET backbone loaded")
        
#     elif pretrain_data == "coco":      
#         weights = DeepLabV3_ResNet50_Weights.DEFAULT
#         coco_backbone = deeplabv3_resnet50(weights=weights).backbone
#         coco_state_dict = get_state_dict(coco_backbone)
#         resnet50_backbone.load_state_dict(coco_state_dict, strict=False)
#         logger.info("COCO backbone loaded")

#     elif pretrain_data == "sup":
#         logger.info("No model loaded!")
        
#     elif pretrain_data == "seg":
#         model = DeepLabV3(
#             backbone=resnet50_backbone,
#             learning_rate=learning_rate,
#             num_classes=num_classes,
#             freeze_backbone=False
#         )

#         resnet50_backbone = FromPretrained(
#             model=model, ckpt_path=import_path, strict=False, error_on_missing_keys=False
#         ).backbone
        
#         logger.info("Default model loaded (seg)")
        
#     elif pretrain_data == "teste":
#         resnet50_backbone = deeplabv3_resnet50().backbone
#         logger.info("Imported backbone from scratch loaded")
    
#     else:
#         raise KeyError("Pretrain data value wrong!")

#     # Build final model
#     if linear:
#         pred_head = LinearSegmentationHead(
#             in_channels=2048,
#             num_classes=num_classes
#         )

#         model = DeepLabV3(
#             backbone=resnet50_backbone,
#             pred_head=pred_head,
#             learning_rate=learning_rate,
#             num_classes=num_classes,
#             freeze_backbone=freeze
#         )
        
#     else:
#         model = DeepLabV3(
#             backbone=resnet50_backbone,
#             learning_rate=learning_rate,
#             num_classes=num_classes,
#             freeze_backbone=freeze
#         )

#     return model

import json
import os

def get_model(
    pretrain_data,
    learning_rate,
    repetition,
    ckpt_path_override=None,
    finetune_data=None,
    freeze=False,
    linear=False,
    config_path="pretrain_configs.json"
):
    num_classes = 6

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
        cfg = all_configs.get(pretrain_data, all_configs["default"])
    else:
        logger.warning(f"Config {config_path} não encontrado. Usando padrões hardcoded.")
        cfg = {"in": 256, "batch_size": 32, "num_epochs": 1200, "lr": "1e-05"}

    if ckpt_path_override:
        import_path = ckpt_path_override
    elif pretrain_data == "seg":
        seg_map = {"f3_N": "seam_ai_N", "f3": "seam_ai", "seam_ai_N": "f3_N", "seam_ai": "f3"}
        target_dataset = seg_map[finetune_data]
        root_seg = "./checkpoints/ckpt_vinicius/train"
        import_path = get_seg_file(root_path=root_seg, repetition=repetition, seg_data=target_dataset)
    else:
        model_name = f"V{repetition}_pretrain_{pretrain_data}_In{cfg['in']}_B{cfg['batch_size']}_E{cfg['num_epochs']}_lr{cfg['lr']}"
        import_path = f"./checkpoints/ckpt_vinicius/pretrain/{repetition}/{model_name}/{pretrain_data}/last.ckpt"


    resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)
    byol_list = ["f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700", "namss"]
    
    if pretrain_data in byol_list or pretrain_data == "seg":
        model_wrapper = BYOL(backbone=resnet50_backbone, learning_rate=learning_rate) 
        resnet50_backbone = FromPretrained(
            model=model_wrapper, 
            ckpt_path=import_path, 
            strict=False
        ).backbone 
    if linear:
        pred_head = LinearSegmentationHead(in_channels=2048, num_classes=num_classes)
        model = DeepLabV3(backbone=resnet50_backbone, pred_head=pred_head, learning_rate=learning_rate, num_classes=num_classes, freeze_backbone=freeze)
    else:
        model = DeepLabV3(backbone=resnet50_backbone, learning_rate=learning_rate, num_classes=num_classes, freeze_backbone=freeze) 

    return model



def new_get_model(
        pretrain_data,
        learning_rate,
        freeze,
        repetition,
        root_path=None,
        full_path=False,
        linear=False,
        finetune_data=None,
        lr_config=None,   # <--- NEW ARG
    ):
    num_classes = 6

    if full_path: 
        import_path = full_path
    else:
        if pretrain_data == "a700":
            base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B128_E125000_lr1e-05"
        elif pretrain_data == "namss":
            base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B128_E125000_lr1e-05"
        else:
            base_name = f"V{repetition}_pretrain_{pretrain_data}_In256_B32_E1200_lr1e-05"

        if root_path:
            import_path = f"{root_path}/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
        else:
            import_path = (
                f"ckpt/pretrain/{repetition}/{base_name}/{pretrain_data}/last.ckpt"
            )
            
    if pretrain_data == "seg":
        seg_data_mapping = {
            "f3_N": "seam_ai_N",
            "f3": "seam_ai",
            "seam_ai_N": "f3_N",
            "seam_ai": "f3",
        }
        root_path = "/petrobr/parceirosbr/home/vinicius.soares/workspace/Seismic-Byol/dev-seismic-byol/checkpoints/ckpt_vinicius/train"
        if finetune_data in ["f3_N", "seam_ai_N", "f3", "seam_ai"]:
            import_path = get_seg_file(
                root_path=root_path,
                repetition=repetition,
                seg_data=seg_data_mapping[finetune_data],
            )
            logger.info(f"CKPT imported from:\n{import_path}")
            logger.info(f"Imported pretrain for segmentation in {seg_data_mapping[finetune_data]}")
        else:
            raise ValueError("Invalid finetune_data for pretrain_data 'seg'")

    seg_data = ["f3", "f3_N", "seam_ai", "seam_ai_N", "both", "both_N", "s0", "a700", "namss"]

    resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)

    if pretrain_data in seg_data:
        # Builds the BYOL model, loads the weights and extracts the backbone
        model = BYOL(backbone=resnet50_backbone, learning_rate=learning_rate)
        logger.info(import_path)
        resnet50_backbone = FromPretrained(
            model=model,
            ckpt_path=import_path,
            strict=False,
            error_on_missing_keys=False,
        ).backbone
        logger.info(f"{pretrain_data} backbone loaded")

    elif pretrain_data == "imagenet":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        imagenet_backbone = resnet50(
            replace_stride_with_dilation=[False, True, True], weights=weights
        )
        imagenet_state_dict = get_state_dict(imagenet_backbone)
        resnet50_backbone.load_state_dict(imagenet_state_dict, strict=False)
        logger.info("IMAGENET backbone loaded")
        
    elif pretrain_data == "coco":      
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        coco_backbone = deeplabv3_resnet50(weights=weights).backbone
        coco_state_dict = get_state_dict(coco_backbone)
        resnet50_backbone.load_state_dict(coco_state_dict, strict=False)
        logger.info("COCO backbone loaded")

    elif pretrain_data == "sup":
        logger.info("No model loaded!")
        
    elif pretrain_data == "seg":
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
        )

        resnet50_backbone = FromPretrained(
            model=model, ckpt_path=import_path, strict=False, error_on_missing_keys=False
        ).backbone
        
        logger.info("Default model loaded (seg)")
        
    elif pretrain_data == "teste":
        resnet50_backbone = deeplabv3_resnet50().backbone
        logger.info("Imported backbone from scratch loaded")
    
    else:
        raise KeyError("Pretrain data value wrong!")

    # Build final model
    if linear:
        pred_head = LinearSegmentationHead(
            in_channels=2048,
            num_classes=num_classes
        )

        model = DeepLabV3(
            backbone=resnet50_backbone,
            pred_head=pred_head,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=freeze
        )
        
    else:
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=freeze
        )
    
    if lr_config is not None:
        if isinstance(lr_config, dict):
            logger.info(f"Applying custom LR per block: {lr_config}")
            model.backbone, param_groups = apply_layerwise_lr(model.backbone, lr_config)
            model.param_groups = param_groups  # store for optimizer configuration
        else:
            raise TypeError("`lr_config` must be a dict mapping layer names to learning rates.")
    elif freeze:
        # Default: freeze everything
        logger.info("Freezing entire backbone (all layers).")
        model.backbone, _ = apply_layerwise_lr(model.backbone, {
            "stem": 0.0, "layer1": 0.0, "layer2": 0.0, "layer3": 0.0, "layer4": 0.0, "avgpool": 0.0
        })

    return model


def get_eval_model(pretrain_data, import_path, learning_rate, linear=False):

    num_classes = 6

    seg_data = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N",
        "both",
        "both_N",
        "s0",
        "a700",
        "sup",
        "seg",
        "coco",
        "imagenet",
        "seg",
        "namss"
    ]

    if pretrain_data in seg_data:
        resnet50_backbone = DeepLabV3Backbone(num_classes=num_classes)
        
    elif pretrain_data == "teste":
        resnet50_backbone = deeplabv3_resnet50().backbone
        logger.info("Imported backbone from sratch loaded")

    else:
        raise KeyError("Pretrain data value wrong!")
    
    if linear:
        pred_head = LinearSegmentationHead(
            in_channels=2048,
            num_classes=num_classes
        )

        model = DeepLabV3(
            backbone=resnet50_backbone,
            pred_head=pred_head,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
        )
        
    else:
        model = DeepLabV3(
            backbone=resnet50_backbone,
            learning_rate=learning_rate,
            num_classes=num_classes,
            freeze_backbone=False
        )

    model = FromPretrained(
        model=model, ckpt_path=import_path, strict=False, error_on_missing_keys=False
    )

    return model
