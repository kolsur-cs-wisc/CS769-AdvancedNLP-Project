import json
import torch
import os
from copy import deepcopy
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from data.slake_datasetV2 import SLAKE_Dataset
    
def create_ans2label(dataset_train, dataset_validate, dataset_test):
    samples = deepcopy(dataset_train)
    samples.extend(dataset_validate)
    samples.extend(dataset_test)
    possible_answers = sorted(set([sample['answer'].lower() for sample in samples]))
    ans_to_label = {}
    label_to_ans = {}
    for i in range(len(possible_answers)):
        label_to_ans[i] = possible_answers[i]
        ans_to_label[possible_answers[i]] = i
    return label_to_ans, ans_to_label

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

f = open('root_slake/train.json')
train_json = json.load(f)

f = open('root_slake/test.json')
test_json = json.load(f)

f = open('root_slake/validate.json')
validate_json = json.load(f)

label_to_ans, ans_to_label = create_ans2label(train_json, validate_json, test_json)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

train_dataset = SLAKE_Dataset(data = train_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)
test_dataset = SLAKE_Dataset(data = test_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)
validate_dataset = SLAKE_Dataset(data = validate_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", id2label=label_to_ans, label2id=ans_to_label)
model.classifier[3] = torch.nn.Linear(in_features=1536 , out_features=36, bias=True)
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=24, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

model.train()
for epoch in range(5):  # loop over the dataset multiple times
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

    for batch in tqdm(validate_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        predicted_classes = torch.sigmoid(logits)
        probs, classes = torch.topk(predicted_classes, 1)

        true_labels = torch.tensor(batch['labels'].nonzero(as_tuple=True)[1])

        correct += (classes[1] == true_labels).float().sum()
    print("Validate Accuracy: ", correct/len(validate_dataset))

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
