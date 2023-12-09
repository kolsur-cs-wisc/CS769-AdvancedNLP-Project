# Train New Models

## Pretraining
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=<BS_FITS_YOUR_GPU>

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_mlm_itm whole_word_masking=True step200k per_gpu_batchsize=64
```

## Finetune on NLVR2
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_nlvr2_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_nlvr2_randaug per_gpu_batchsize=32 load_path="weights/vilt_200k_mlm_itm.ckpt"
```

## Finetune on VQAv2
```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_trainval_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path="<YOUR_WEIGHT_ROOT>/vilt_200k_mlm_itm.ckpt"

ex)
python run.py with data_root=/data2/dsets/dataset num_gpus=8 num_nodes=1 task_finetune_vqa_randaug per_gpu_batchsize=64 load_path="weights/vilt_200k_mlm_itm.ckpt"
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