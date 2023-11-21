from vilt.datasets import SLAKEDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict

class SLAKEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SLAKEDataset

    @property
    def dataset_name(self):
        return "slake"