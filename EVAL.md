# Evaluation
The results will vary a bit since we do a batched-inference, which yields padded image batch that would be inconsistently embedded while performing linear image patch projection.

## Evaluate VQAv2
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_vqa_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_vqa.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=64 task_finetune_vqa_randaug test_only=True precision=32 load_path="weights/vilt_vqa.ckpt"

output > This script will generate `result/vqa_submit_vilt_vqa.json`, you can upload it to eval.ai (https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get test-dev score.
[{"test-dev": {"yes/no": 87.44, "number": 50.2, "other": 62.38, "overall": 71.32}}]
```

## Evaluate NLVR2
```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> per_gpu_batchsize=<BS_FITS_YOUR_GPU> task_finetune_nlvr2_randaug test_only=True precision=32 load_path="<YOUR_WEIGHT_ROOT>/vilt_nlvr2.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 per_gpu_batchsize=64 task_finetune_nlvr2_randaug test_only=True precision=32 load_path="weights/vilt_nlvr2.ckpt"

output >
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'nlvr2/dev/accuracy': tensor(0.7486, device='cuda:0'),
 'nlvr2/dev/accuracy_epoch': tensor(0.7565, device='cuda:0'),
 'nlvr2/dev/loss': tensor(0.8581, device='cuda:0'),
 'nlvr2/dev/loss_epoch': tensor(0.8609, device='cuda:0'),
 'nlvr2/test/accuracy': tensor(0.7735, device='cuda:0'),
 'nlvr2/test/accuracy_epoch': tensor(0.7652, device='cuda:0'),
 'nlvr2/test/loss': tensor(0.7796, device='cuda:0'),
 'nlvr2/test/loss_epoch': tensor(0.8381, device='cuda:0'),
 'val/the_metric': tensor(0.7652, device='cuda:0')}
--------------------------------------------------------------------------------
INFO - ViLT - Completed after 0:01:31
```
# Train & Evaluate Bio Medical Datasets
Run all commands mentioned below from the main project folder. 

## Finetune Base ViLT on SLAKE Dataset
```python finetune_slake.py```

## Finetune Base ViLT on VQA RAD Dataset
```python finetune_vqarad.py```

## Finetune LoRA Adapated ViLT on SLAKE Dataset
```python finetune_slake_lora.py```

## Finetune LoRA Adapated ViLT on VQA RAD Dataset
```python finetune_slake_vqarad.py```

# Results
All our results have also been added to the ```results``` folder under various folders. If you're seeing any errors please refer to those for expected results obtained.
