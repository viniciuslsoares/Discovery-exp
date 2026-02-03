# cli.py

import argparse
from pretrain import main
from functions import *
from auxiliar import *
from data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretraining script for seismic segmentation with BYOL."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="Input size (used for both height and width)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="seam_ai", help="Name of the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--repetition", type=int, default=0, help="Repetition being run"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Models learning rate"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU device indices to use",
    )

    args = parser.parse_args()

    dataset_mapping = get_dataset_mapping()
    
    logger.info(" =-=-=- Begining training =-=-=-")

    logger.info(f"Dataset path: {dataset_mapping[args.dataset_name]}")

    pretrain_list = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N",
        "both",
        "both_N",
        "s0",
        "a700",
        "imagenet",
        "coco",
        "sup",
        "namss",
    ]

    if args.dataset_name not in pretrain_list:
        raise KeyError(
            f"Pretrain '{args.dataset_name}' not found in available options: {pretrain_list}"
        )

    PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/pretrain/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/pretrain/{args.repetition}"

    logger.info(f"Batches: {args.batch_size} - Input: {args.input_size}")
    logger.info(f"Pretrain Log Path: {PRETRAIN_LOGS_PATH}")
    logger.info(f"Pretrain CKPT Path: {PRETRAIN_CKPT_PATH}")

    main(
        input_size=(args.input_size, args.input_size),
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        repetition=args.repetition,
        learning_rate=args.learning_rate,
        data_path=dataset_mapping[args.dataset_name],
        log_path=PRETRAIN_LOGS_PATH,
        ckpt_path=PRETRAIN_CKPT_PATH,
        gpus=args.gpus,
    )
