from pathlib import Path

from functions import *
from auxiliar import *
from data import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import Accuracy, JaccardIndex, F1Score


def main(
    pretrain_data,  # Where the model was pretrained
    finetune_data,  # Where the model is going to be trained
    data_path,      # Root path for the data
    num_epochs,     # Epochs to be trained
    batch_size,     
    repetition,     # Repetition and random seed
    learning_rate,  
    cap,            # Amount of samples to be used. 1.0 means 100%
    freeze,         # Freeze backbone
    ckpt_path,      # Where to save the checkpoints
    logs_path,      # Where to save the logs
    gpus,           # Gpus to run       
    full_save_name=None,   # If want to save as a specific name
    linear=False,   # FC to choose from linear or default
    steps=False,    # If num_epochs refer to steps
    pretrain_config="pretrain_configs.json", 
    ckpt_path_override=None,                 
):

    seed_everything(repetition)
    if full_save_name:
        save_name = full_save_name
    else:
        if isinstance(cap, float):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap*100:.0f}%"
            )
        elif isinstance(cap, int):
            save_name = (
                f"V{repetition}_pre_{pretrain_data}_train_{finetune_data}_cap_{cap}_img"
            )
        
    logger.info(f"Saving model {save_name}")
    
    # Transforms
    if finetune_data == "f3" or finetune_data == "f3_N":
        logger.info("Using padding of (256,704)")
        padding = Padding(256, 704)
    elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
        logger.info("Using padding of (1008,592)")
        padding = Padding(1008, 592)
        
    crop = RandomCrop(
        crop_size=(224, 224),
        num_samples=2
        )
    
    transform_pipeline = TransformPipeline([
        padding,
        Transpose([2, 0, 1]),   # C, H, W

    ])
    
    if cap == 1.0 and type(cap) == float:
        
        logger.info("Using 100% of train data")
        train_dataset = SeismicFullDataset(root=data_path, partition='train', transform=transform_pipeline)
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=True,
            transform=transform_pipeline,
            test_transform=transform_pipeline,
            train_dataset = train_dataset,
            val_dataset = None,
            test_dataset = None,
        )
        
    else:
        logger.info(f'Using {cap} samples of train data')
        data_module = SeismicDataModule(
            root = data_path,
            batch_size=batch_size,
            cap=cap,
            drop_last=False,
            transform=transform_pipeline,
            test_transform=transform_pipeline,
            train_dataset = None,
            val_dataset = None,
            test_dataset = None,
        )

    model = get_model(
        pretrain_data=pretrain_data,
        learning_rate=learning_rate,
        freeze=freeze,
        repetition=repetition,
        finetune_data=finetune_data,
        linear=linear,
        config_path=pretrain_config,
        ckpt_path_override=ckpt_path_override
    )

    log_dir = Path(logs_path) / save_name / finetune_data
    ckpt_dir = Path(ckpt_path) / save_name / finetune_data
    csv_logger = CSVLogger(log_dir, name=save_name, version=finetune_data)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, dirpath=ckpt_dir, mode="min", monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
    )

    if steps:
        trainer = Trainer(
            accelerator="gpu",
            logger=csv_logger,
            callbacks=[ckpt_callback, early_stopping_callback],
            max_steps=num_epochs,
            strategy="auto",
            devices=gpus,
            check_val_every_n_epoch=True,
        )

    else:

        trainer = Trainer(
            accelerator="gpu",
            logger=csv_logger,
            callbacks=[ckpt_callback, early_stopping_callback],
            max_epochs=num_epochs,
            strategy="auto",
            devices=gpus,
            check_val_every_n_epoch=True,
        )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
    )

    pipeline.run(data_module, task="fit")
    
    num_classes = 6
    
    metrics = {
        "mIoU": JaccardIndex(
            num_classes=num_classes, average="macro", task="multiclass"
        ),
        "acc": Accuracy(num_classes=num_classes, task="multiclass"),
        "f1-weighted": F1Score(
            num_classes=num_classes, task="multiclass", average="weighted"
        ),
    }
    
    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
        seed=repetition,
        apply_metrics_per_sample=False,
        classification_metrics=metrics,
    )
    
    pipeline.run(data_module, task="evaluate")

 