# main.py
import os
import numpy as np
from pathlib import Path

from functions import *
from auxiliar import *
from data import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *
from torchvision.models.segmentation import deeplabv3_resnet50
from lightning.pytorch.strategies import DDPStrategy

from minerva.data.readers import TiffReader
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.fabric import seed_everything

# from a700 import *
# from namss import *

def main(
    input_size,
    dataset_name,
    batch_size,
    num_epochs,
    repetition,
    learning_rate,
    data_path,
    ckpt_path,
    log_path,
    gpus,
):
    
    seed_everything(repetition)
    model_name = f'V{repetition}_pretrain_{dataset_name}_In{str(input_size[0])}_B{batch_size}_E{num_epochs}_lr{learning_rate}'
    logger.info(f'Model name: {model_name}')
    
    if dataset_name == 'a700' or dataset_name == 'namss':
        aux_transform_pipeline = TransformPipeline(
            [
                RandomFlip(possible_axis=[0, 1]),   # 513, 513
                RandomRotation(degrees=25, prob=0.2),
                Unsqueeze(axis=0),  # 1, 513, 513
                Repeat(axis=0, n_repetitions=3),    # 3, 513, 513
                CastTo(dtype=np.float32),
            ]
        )

        constrastive_transform = ContrastiveTransform(
            aux_transform_pipeline
        )

        byol_transform_pipeline = TransformPipeline(
            [
                RandomCrop(crop_size=input_size),    # 3, crop_size, crop_size
                constrastive_transform,
            ]
        )   
    
        
    else: 
        aux_transform_pipeline = TransformPipeline(
            [   
                RandomFlip(possible_axis=[1, 2]),        # H, W, C
                RandomRotation(degrees=25, prob=0.2),
                Transpose([2, 0, 1]),   # C, H, W
                CastTo(dtype=np.float32),
            ]
        ) 

        constrastive_transform = ContrastiveTransform(
            aux_transform_pipeline
        )

        byol_transform_pipeline = TransformPipeline(
            [
                RandomCrop(crop_size=input_size), # 3, crop_size, crop_size
                constrastive_transform,
            ]
        )
        
    logger.info(f"Transforms built for {dataset_name}")
    
    if dataset_name == 'a700':
        data_module = A150DataModule(
            root=data_path,
            subset='both',
            batch_size=batch_size,
            transforms=j,
            num_workers=os.cpu_count() if os.cpu_count() < 24 else 24,
            drop_last=True,
        )
    
    elif dataset_name == 'namss':
        data_module = NAMSSDataModule(
            root_path=data_path,
            batch_size=batch_size,
            num_workers=os.cpu_count() if os.cpu_count() < 24 else 24 ,
            drop_last=True,
            transforms=byol_transform_pipeline,    
        )

    else:
    
        train_img_reader = TiffReader(path=data_path)
        logger.info(f"Readers built!")
       
        pretrain_dataset = SimpleDataset(
            readers=train_img_reader,
            transforms=byol_transform_pipeline,
            return_single=True
        )
        logger.info(f"Dataset built!")

        
        data_module = MinervaDataModule(
            train_dataset=pretrain_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle_train=True,
            name=dataset_name,
            num_workers=os.cpu_count()
        )
        

    logger.info(f'DataModule assembled')

    # Modelo
    backbone = DeepLabV3Backbone(num_classes=6)
    model = BYOL(backbone=backbone, 
                 learning_rate=learning_rate,
                 )
    logger.info(f"Model built: {type(model).__name__}")
    # Logger, Checkpoints, Trainer
    log_dir = Path(log_path) / model_name / dataset_name
    ckpt_dir = Path(ckpt_path) / model_name / dataset_name
    CSVlogger = CSVLogger(log_dir, name=model_name, version=dataset_name)
    
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, 
        save_last=True, 
        dirpath=ckpt_dir,
        )
    
    logger.info("Loggers and checkpoints built")

    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        logger=CSVlogger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        # max_steps=num_epochs,
        strategy='auto',
        # strategy=DDPStrategy(static_graph=True),
        log_every_n_steps=30,
    )
    logger.info("Trainer instantiated")

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")
    