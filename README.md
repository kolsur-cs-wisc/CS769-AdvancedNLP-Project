# CS769 Advanced NLP Project ViLT 2.0: Parameter Efficient Fine Tuned ViLT for Biomedical VQA

Code for the ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision paper and CS769 Project ViLT 2.0: Parameter Efficient Fine Tuned ViLT for Biomedical VQA.

## Install
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

## Download Pretrained Weights
ViLT-B/32 Pretrained with MLM+ITM for 200k steps on GCC+SBU+COCO+VG (ViLT-B/32 200k) [link](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)

## Dataset Preparation
See [`DATA.md`](DATA.md)

## Train New Models
See [`TRAIN.md`](TRAIN.md)

## Evaluation
See [`EVAL.md`](EVAL.md)