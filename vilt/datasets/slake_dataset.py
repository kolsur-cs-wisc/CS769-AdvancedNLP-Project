from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split

        if split == "train":
            names = ["train"]
        elif split == "val":
            names = ["validate"]
        elif split == "test":
            names = ["test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )