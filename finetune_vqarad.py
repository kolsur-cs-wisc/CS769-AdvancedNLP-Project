import pandas as pd
import json
import torch
import torch.nn.functional as functional
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from data.vqarad_dataset import VQA_RAD_Dataset

def create_ans2label(dataset_train, dataset_test, vocab):
   vocab_unique = vocab['Medical Words'].str.lower().unique()
   unique_train_answers = dataset_train['answer'].str.lower().unique()
   unique_test_answers = dataset_test['answer'].str.lower().unique()

   train_separated = [word for phrase in unique_train_answers for word in str(phrase).split(' ')]
   test_separated = [word for phrase in unique_test_answers for word in str(phrase).split(' ')]
   
   combined = set(list(train_separated) + list(test_separated) + list(vocab_unique))
   unique_answers = list(combined)

   answer_to_index = {answer: index for index, answer in enumerate(unique_answers)}
   one_hot_vectors = [functional.one_hot(torch.tensor(answer_to_index[answer]), len(unique_answers)) for answer in unique_answers]

   predicted_answers = [unique_answers[tensor.argmax().item()] for tensor in one_hot_vectors]

   return unique_answers, answer_to_index, predicted_answers
   

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

f = open('root_vqarad/tokens.json')
tokens_json = json.load(f)

unique_answers = tokens_json['unique_answers']
answer_to_index = tokens_json['answer_to_index']
predicted_answers = tokens_json['predicted_answers']

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

vocab = pd.read_csv('root_vqarad/medical-vocab.csv')
train_data = pd.read_csv('root_vqarad/train-dataV2.csv')
test_data = pd.read_csv('root_vqarad/test-dataV2.csv')

unique_answers, answer_to_index, predicted_answers = create_ans2label(train_data, test_data, vocab)

train_dataset = VQA_RAD_Dataset(data = train_data, processor = processor, ans2label = answer_to_index)
test_dataset = VQA_RAD_Dataset(data = test_data, processor = processor, ans2label = answer_to_index)

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm")
model.classifier[3] = torch.nn.Linear(in_features=1536 , out_features=1296, bias=True)
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

model.train()
for epoch in range(3):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")

    for batch in tqdm(train_dataloader):
      batch = {k:v.to(device) for k,v in batch.items()}
      optimizer.zero_grad()
      
      outputs = model(**batch)
      loss = outputs.loss
      print("Loss:", loss.item())
      loss.backward()
      optimizer.step()

correct = 0
for batch in tqdm(test_dataloader):
    batch = {k:v.to(device) for k,v in batch.items()}

    outputs = model(**batch)
    logits = outputs.logits

    predicted_classes = torch.sigmoid(logits)
    probs, classes = torch.topk(predicted_classes, 1)

    true_labels = torch.tensor(batch['labels'].nonzero(as_tuple=True)[1])

    correct += (classes[1] == true_labels).float().sum()
print("Test Accuracy: ", correct/len(test_dataset))