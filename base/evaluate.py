from pathlib import Path

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PNGReader
from minerva.data.datasets import SimpleDataset

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from torchmetrics import Accuracy, JaccardIndex, F1Score
from lightning.fabric import seed_everything


def main(
    ckpt_file,
    model_name,
    finetune_data,
    pretrain_data,
    data_path,
    num_epochs,
    batch_size,
    repetition,
    ckpt_path,
    logs_path,
    gpus,
    linear,
):

    # Set general seed
    seed_everything(repetition)

    # Transforms
    if finetune_data == "f3" or finetune_data == "f3_N":
        padding = Padding(256, 704, padding_mode='reflect')
    elif finetune_data == "seam_ai" or finetune_data == "seam_ai_N":
        padding = Padding(1008, 592, padding_mode='reflect')
        
    transform_pipeline = TransformPipeline(
        [
            padding,
            Transpose([2, 0, 1]),   # C, H, W
            CastTo(dtype=np.float32),
        ]
    )

    # DataModule
    data_module = SeismicDataModule(
        root = data_path,
        batch_size=batch_size,
        cap=1.0,
        drop_last=True,
        transform=transform_pipeline,
        test_transform=transform_pipeline,
        train_dataset = None,
        val_dataset = None,
        test_dataset = None,
    )

    model = get_eval_model(
        pretrain_data=pretrain_data,
        import_path=ckpt_file,
        learning_rate=0.001,
        linear=linear
        )

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

    log_dir = Path(logs_path) / model_name
    ckpt_dir = Path(ckpt_path) / model_name 
    logger = CSVLogger(log_dir, model_name, version=finetune_data)
    ckpt_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, dirpath=ckpt_dir, mode="min", monitor="val_loss"
    )

    # trainer = Trainer(
    #     accelerator="cpu",
    #     logger=logger,
    #     callbacks=[ckpt_callback],
    #     max_epochs=num_epochs,
    #     strategy="auto",
    #     check_val_every_n_epoch=2,
    # )

    trainer = Trainer(
        accelerator="gpu",
        logger=logger,
        callbacks=[ckpt_callback],
        max_epochs=num_epochs,
        strategy="auto",
        devices=[0],
        check_val_every_n_epoch=2,
    )

    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
        seed=repetition,
        apply_metrics_per_sample=False,
        classification_metrics=metrics,
    )

    # @torch.no_grad()
    pipeline.run(data_module, task="evaluate")

if __name__ == "__main__":
    main(
        model_name="V0_pre_a700_train_f3_N_cap_100%",
        ckpt_file="/home/vinicius.soares/Seismic-Byol/dev-seismic-byol/ckpt/train/0/V0_pre_a700_train_f3_N_cap_100%/f3_N/epoch=49-step=6200.ckpt",
        pretrain_data="a700",
        finetune_data="f3_N",
        data_path='/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
        num_epochs=50,
        batch_size=8,
        repetition=0,
        ckpt_path="ckpt/test/0",
        logs_path="logs/test/0",
        gpus=[0],
        linear=False,
    )




