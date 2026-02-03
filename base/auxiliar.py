import logging

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = get_logger("minerva")

def extract_epoch_number(filename):
    match = re.search(r"epoch=(\d+)", filename)
    return int(match.group(1)) if match else -1

from pathlib import Path
import re

def get_models_files(base_dir="./ckpt/train", target_repetition=None):
    base_dir = Path(base_dir)
    results = []
    repetitions = (
        [str(target_repetition)]
        if target_repetition != None
        else [d.name for d in base_dir.iterdir() if d.is_dir()]
    )

    for repetition_dir in repetitions:
        rep_path = base_dir / repetition_dir
        if not rep_path.is_dir():
            continue

        for model_dir in rep_path.iterdir():
            if not model_dir.is_dir():
                continue

            match = re.match(r"V(\d+)_pre_(.+?)_train_(.+?)_cap_(.+)", model_dir.name)
            if not match:
                continue

            _, pretrain_data, train_data, cap = match.groups()

            for train_data_dir in model_dir.iterdir():
                if not train_data_dir.is_dir():
                    continue

                ckpt_files = [
                    f
                    for f in train_data_dir.iterdir()
                    if f.is_file() and f.name.startswith("epoch=")
                ]
                if ckpt_files:
                    ckpt_files.sort(
                        key=lambda f: extract_epoch_number(f.name), reverse=True
                    )
                    results.append(
                        {
                            "model_name": model_dir.name,
                            "repetition": repetition_dir,
                            "pretrain_data": pretrain_data,
                            "train_data": train_data,
                            "cap": cap,
                            "ckpt_file": str(ckpt_files[0]),
                        }
                    )

    return results

def parse_cap(value):
    try:
        if '.' in value:
            val = float(value)
            if not (0.0 < val <= 1.0):
                raise argparse.ArgumentTypeError("Float cap must be in the (0, 1] range.")
            return val
        else:
            val = int(value)
            if val <= 0:
                raise argparse.ArgumentTypeError("Integer cap must be greater than 0.")
            return val
    except ValueError:
        raise argparse.ArgumentTypeError("Cap must be a float in (0, 1] or a positive int.")
    
    
def parse_freeze_list(values):
    # Se o argparse passou como lista (nargs=5)
    if isinstance(values, list):
        raw_list = values
    else:
        # Se veio como string única
        raw_list = values.replace(",", " ").split()

    try:
        lst = [int(v) for v in raw_list]
        if len(lst) != 5:
            raise argparse.ArgumentTypeError("freeze_list must contain exactly 5 values (0 or 1).")
        if any(v not in (0, 1) for v in lst):
            raise argparse.ArgumentTypeError("freeze_list values must be either 0 or 1.")
        return [bool(v) for v in lst]  # converte 0/1 → False/True
    except ValueError:
        raise argparse.ArgumentTypeError("freeze_list must be a list of integers (0 or 1).")