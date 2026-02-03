import os
import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import lightning as L

from minerva.data.data_modules.base import MinervaDataModule
from minerva.data.datasets.binary_tree_subset import BinaryTreeSubset
from minerva.data.datasets.base import SimpleDataset
from minerva.data.readers import TiffReader, PNGReader
from auxiliar import logger


def apply_squeeze(label):
    """Remove dimensões unitárias (ex: [1, H, W] -> [H, W])"""
    if torch.is_tensor(label):
        return label.squeeze()
    elif isinstance(label, np.ndarray):
        return np.squeeze(label)
    return label


class SeismicReducibleDataset(Dataset):
    def __init__(self, root: Path, size: int, transform = None):
        assert size > 0, f"`size` must be a positive integer, but got size = {size}"
        self.root = Path(root)

        xl_set = SimpleDataset([
            TiffReader(self.root / "images/train", ["text", "numeric"], "_", [0, 1], False, r"xl.*"),
            PNGReader(self.root / "annotations/train", ["text", "numeric"], "_", [0, 1], False, r"xl.*"),
        ])

        il_set = SimpleDataset([
            TiffReader(self.root / "images/train", ["text", "numeric"], "_", [0, 1], False, r"il.*"),
            PNGReader(self.root / "annotations/train", ["text", "numeric"], "_", [0, 1], False, r"il.*"),
        ])

        max_size = len(xl_set) + len(il_set)
        assert max_size >= size, f"There are only {max_size} samples in the dataset but got size = {size}"

        xl_size = min(size // 2, len(xl_set))
        il_size = min(size - xl_size, len(il_set))

        sets = []
        if xl_size > 0: sets.append(BinaryTreeSubset(xl_set, xl_size))
        if il_size > 0: sets.append(BinaryTreeSubset(il_set, il_size))

        self.data: Dataset = ConcatDataset(sets)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index]
        # Aplica transformações e remove dimensão de canal da label
        return self.transform(image), apply_squeeze(self.transform(label))

    def __len__(self):
        return len(self.data)


class SeismicFullDataset(SimpleDataset):
    def __init__(self, root: Path, partition: Literal["val", "test", "train"], transform):
        self.root = Path(root)
        super().__init__(
            [
                TiffReader(self.root / f"images/{partition}"),
                PNGReader(self.root / f"annotations/{partition}"),
            ],
            transforms=transform
        )

    def __getitem__(self, index):
        # Sobrescreve o __getitem__ da SimpleDataset para aplicar o squeeze
        image, label = super().__getitem__(index)
        return image, apply_squeeze(label)


class SeismicDataModule(MinervaDataModule):
    def __init__(
        self,
        root: Path,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() if os.cpu_count() < 24 else 24,
        cap: int = 256,
        drop_last: bool = False,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        transform=None,
        test_transform=None,
        *args,
        **kwargs,
    ):
        # Defina os datasets caso não tenham sido passados
        train_dataset = train_dataset or SeismicReducibleDataset(
            root=root, size=cap, transform=transform
        )
        val_dataset = val_dataset or SeismicFullDataset(
            root=root, partition="val", transform=test_transform
        )
        test_dataset = test_dataset or SeismicFullDataset(
            root=root, partition="test", transform=test_transform
        )

        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            *args,
            **kwargs,
            additional_train_dataloader_kwargs={'pin_memory':True},
            additional_val_dataloader_kwargs={'pin_memory':True}, 
            additional_test_dataloader_kwargs={'pin_memory':True},
        )



class CapDataModule(MinervaDataModule):
    def __init__(
        self,
        cap_train: Optional[Union[float, int]] = None,
        cap_val: Optional[Union[float, int]] = None,
        cap_test: Optional[Union[float, int]] = None,
        seed: Optional[int] = 42,
        drop_last: Optional[bool] = False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cap_train = cap_train
        self.cap_val = cap_val
        self.cap_test = cap_test
        self.seed = seed
        self.drop_last = drop_last
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _get_capped_dataloader(self, dataloader, cap_value, shuffle=False):
        if cap_value is None:
            return dataloader
            
        if isinstance(cap_value, float):
            cap_len = int(len(dataloader.dataset) * cap_value)
            subset, _ = torch.utils.data.random_split(
                dataloader.dataset,
                [cap_len, len(dataloader.dataset) - cap_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
        elif isinstance(cap_value, int):
            subset = BinaryTreeSubset(dataloader.dataset, cap_value)
        else:
            raise TypeError("Cap must be float or int.")

        return torch.utils.data.DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=shuffle,
            num_workers=15,
            pin_memory=dataloader.pin_memory,
            drop_last=self.drop_last,
        )

    def train_dataloader(self):
        return self._get_capped_dataloader(super().train_dataloader(), self.cap_train, shuffle=True)

    def val_dataloader(self):
        return self._get_capped_dataloader(super().val_dataloader(), self.cap_val, shuffle=False)

    def test_dataloader(self):
        return self._get_capped_dataloader(super().test_dataloader(), self.cap_test, shuffle=False)


import os

def get_dataset_mapping():
    
    nodename = os.uname().nodename
    
    if 'sdumont' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/seam_ai_N',
            'seam_ai':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/seam_ai',
            'f3':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/f3_segmentation',
            'f3_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/f3_segmentation_N',
            'both':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/both',
            'both_N':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/tiff_data/both_N',
            'a700':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/a700',
            'namss':'/petrobr/parceirosbr/home/vinicius.soares/workspace/spfm/datasets/NAMSS/Data/NAMSS/patch_512_0',
        }
    
    elif 'node' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/workspaces/shared_data/seam_ai_datasets/seam_ai_N/images',
            'seam_ai':'/workspaces/shared_data/seam_ai_datasets/seam_ai/images',
            'f3':'/workspaces/shared_data/seismic/f3_segmentation/images',
            'f3_N':'/workspaces/shared_data/seismic/f3_segmentation_N/images',
            'both':'/workspaces/shared_data/seismic/both/images',
            'both_N':'/workspaces/shared_data/seismic/both_N/images',
        }
        
    elif 'c' in nodename:
        dataset_mapping = {
            'seam_ai_N':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai_N',
            'seam_ai':'/home/vinicius.soares/asml/datasets/tiff_data/seam_ai',
            'f3':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation',
            'f3_N':'/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation_N',
            'both':'/home/vinicius.soares/asml/datasets/tiff_data/both',
            'both_N':'/home/vinicius.soares/asml/datasets/tiff_data/both_N',
            'a700':'/parceirosbr/asml/datasets/a700',
        }
    else:
        raise RuntimeError(f"Unsupported nodename '{nodename}'. Unable to determine dataset mapping.")
    
    return dataset_mapping