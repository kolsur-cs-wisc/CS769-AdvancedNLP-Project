from .base_dataset import BaseDataset

class SLAKEDataset(BaseDataset):
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

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()

        return {
            "image": image_tensor,
            "text": text,
            "slake_answer": answers,
            "slake_labels": labels,
            "qid": qid,
        }