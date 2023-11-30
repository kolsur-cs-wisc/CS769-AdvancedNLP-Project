import torch
import os
from PIL import Image

class VQA_RAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, ans2label):
        self.data = data
        self.processor = processor
        self.questions = self.data['question']
        self.image_ids = self.data['image_name']
        self.answers = self.data['answer']
        self.ans2label = ans2label

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        answer = str(self.answers[idx]).lower()
        questions = self.questions[idx]
        image = Image.open(os.path.join(os.getcwd(), 'root_vqarad', 'osfstorage-archive', 'images', self.image_ids[idx]))
    
        encoding = self.processor(image, questions, padding="max_length", truncation=True, return_tensors="pt")

        for k,v in encoding.items():
            encoding[k] = v.squeeze()

        answer_words = answer.split(' ')
        word_count = len(answer_words)
        scores = [1.0]
        if word_count > 1:
            scores = [(1 / word_count)] * word_count

        targets = torch.zeros(len(self.ans2label))
        scores_final = torch.tensor(scores)
        for ans, score in zip(answer_words, scores_final):
            targets[self.ans2label[ans]] = score
        encoding["labels"] = targets

        return encoding