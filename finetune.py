import json
import torch
from PIL import Image
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

f = open('root/train.json')
train_questions = json.load(f)

f = open('root/test.json')
test_questions = json.load(f)

f = open('root/validate.json')
validate_questions = json.load(f)

label_to_ans, ans_to_label = create_ans2label(train_questions, validate_questions, test_questions)

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

dataset = SLAKE_Dataset(data = train_questions, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=label_to_ans,
                                                 label2id=ans_to_label)
model.to(device)

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

train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k, v.shape)
# # labels = torch.nonzero(batch['labels'][0]).squeeze().tolist()
# # print([config.id2label[label] for label in labels])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(50):  # loop over the dataset multiple times
   print(f"Epoch: {epoch}")
   for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()