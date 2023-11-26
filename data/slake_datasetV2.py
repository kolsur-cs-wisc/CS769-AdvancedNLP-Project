import torch
from PIL import Image

class SLAKE_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, ans2label, label2ans):
        self.data = data
        self.processor = processor
        self.ans2label = ans2label
        self.label2ans = label2ans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annotation = self.data[idx]['answer'].lower()
        questions = self.data[idx]
        image = Image.open(f'root/imgs/{self.data[idx]["img_name"]}')
        text = questions['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        targets = torch.zeros(len(self.ans2label))
        targets[self.ans2label[annotation]] = 1
        encoding["labels"] = targets

        return encoding