import json
import torch
from copy import deepcopy
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from data.slake_datasetV2 import SLAKE_Dataset
from peft import LoraConfig, get_peft_model

    
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

f = open('root_slake/train.json')
train_json = json.load(f)

f = open('root_slake/test.json')
test_json = json.load(f)

f = open('root_slake/validate.json')
validate_json = json.load(f)

label_to_ans, ans_to_label = create_ans2label(train_json, validate_json, test_json)

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

train_dataset = SLAKE_Dataset(data = train_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)
test_dataset = SLAKE_Dataset(data = test_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)
validate_dataset = SLAKE_Dataset(data = validate_json, processor = processor, ans2label = ans_to_label, label2ans = label_to_ans)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", id2label=label_to_ans, label2id=ans_to_label)
config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)
lora_model = get_peft_model(model, config)
# for name, param in lora_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)

lora_model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=24, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(lora_model.parameters(), lr=0.0001)

lora_model.train()

for epoch in range(10):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")

    for batch in tqdm(train_dataloader):
      batch = {k:v.to(device) for k,v in batch.items()}
      optimizer.zero_grad()
      
      outputs = lora_model(**batch)
      loss = outputs.loss
      
      loss.backward()
      optimizer.step()
    print("Loss:", loss.item())
    accurate = 0  
    for batch in tqdm(validate_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}

        outputs = lora_model(**batch)
        
        logits = outputs.logits

        predicted_classes = torch.sigmoid(logits)
        # print(predicted_classes)
        # print(predicted_classes.shape)
        probs, classes = torch.topk(predicted_classes, 1)

        true_labels = torch.tensor(batch['labels'].nonzero(as_tuple=True)[1])
        # print(classes.shape)
        # print(true_labels.shape)
        accurate += (classes[1] == true_labels).float().sum()
    print(accurate/len(validate_dataset))
accurate = 0
for batch in tqdm(test_dataloader):
    batch = {k:v.to(device) for k,v in batch.items()}

    outputs = lora_model(**batch)
    
    logits = outputs.logits

    predicted_classes = torch.sigmoid(logits)
    # print(predicted_classes)
    # print(predicted_classes.shape)
    probs, classes = torch.topk(predicted_classes, 1)

    true_labels = torch.tensor(batch['labels'].nonzero(as_tuple=True)[1])
    # print(classes.shape)
    # print(true_labels.shape)
    accurate += (classes[1] == true_labels).float().sum()
print("Test accuracy : " )
print(accurate/len(test_dataset))

