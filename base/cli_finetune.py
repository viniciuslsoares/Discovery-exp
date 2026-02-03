import argparse
from train import main 
from data import get_dataset_mapping
from auxiliar import logger, parse_cap, parse_freeze_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for seismic segmentation using pretrained BYOL backbone."
    )

    parser.add_argument(
        "--pretrain_data",
        type=str,
        required=True,
        default="seam_ai_N",
        help="Dataset used in pretraining (e.g., f3, seam_ai, both)",
    )
    parser.add_argument(
        "--finetune_data",
        type=str,
        required=True,
        default="seg",
        help="Dataset used for fine-tuning (e.g., f3, seam_ai, both)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--repetition", type=int, default=0, help="Experiment repetition index"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--cap",
        type=parse_cap,
        default=10,
        help="Fraction of data to use for training (between 0 and 1)",
    )
    parser.add_argument(
        "--freeze", action="store_true", help="Whether to freeze the encoder backbone"
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="List of GPU indices to use"
    )
    parser.add_argument(
        "--linear", action="store_true", help="If true uses a linear prediction head"
    )
    parser.add_argument(
        "--steps", action="store_true", help="Defines if num_epochs refers to train steps"
    )
    parser.add_argument(
        "--pretrain_config", 
        type=str, 
        default="pretrain_configs.json", 
        help="Arquivo JSON com os parâmetros de cada modelo pré-treinado"
    )
    parser.add_argument(
        "--ckpt_path_override", 
        type=str, 
        default=None, 
        help="Caminho direto para um arquivo .ckpt (ignora o mapeamento automático)"
    )

    args = parser.parse_args()

    PRETRAIN_LOGS_PATH = f"checkpoints/logs_vinicius/train/{args.repetition}" if not args.linear else f"/logs_vinicius/train_linear/{args.repetition}"
    PRETRAIN_CKPT_PATH = f"checkpoints/ckpt_vinicius/train/{args.repetition}" if not args.linear else f"/ckpt_vinicius/train_linear/{args.repetition}"

    dataset_mapping = get_dataset_mapping()
    
    finetune_list = [
        "f3",
        "f3_N",
        "seam_ai",
        "seam_ai_N"
    ]

    if args.finetune_data not in finetune_list:
        raise KeyError(
            f"Dataset '{args.finetune_data}' not found in available options: {finetune_list}"
        )

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
        "seg",
        "namss",
    ]

    if args.pretrain_data not in pretrain_list:
        raise KeyError(
            f"Pretrain '{args.pretrain_data}' not found in available options: {pretrain_list}"
        )

    logger.info(" =-=-=- Beginning fine-tuning =-=-=-")
    logger.info(f"Pretrain data: {args.pretrain_data}")
    logger.info(f"Finetune data: {args.finetune_data}")
    logger.info(f"Batch size: {args.batch_size}, LR: {args.learning_rate}")
    logger.info(f"Cap: {args.cap}, Freeze: {args.freeze}")
    logger.info(f"Cap type: {type(args.cap)}")
    logger.info(f"Prediction head: {'linear' if args.linear == True else 'DLV3'}")

    main(
        pretrain_data=args.pretrain_data,
        finetune_data=args.finetune_data,
        data_path=dataset_mapping[args.finetune_data],
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        repetition=args.repetition,
        learning_rate=args.learning_rate,
        cap=args.cap,
        freeze=args.freeze,
        ckpt_path=PRETRAIN_CKPT_PATH,
        logs_path=PRETRAIN_LOGS_PATH,
        gpus=args.gpus,
        linear=args.linear,
        steps=args.steps,
        pretrain_config=args.pretrain_config,
        ckpt_path_override=args.ckpt_path_override,
    )
