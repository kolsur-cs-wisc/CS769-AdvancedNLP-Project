# Dataset Preparation
We utilize 4 datsets: COCO Captions (COCO), Natural Language for Visual Reasoning 2 (NLVR2), SLAKE, VQA-RAD

Please download the datasets by yourself.
We use `pyarrow` to serialize the datasets, conversion scripts are located in `vilt/utils/write_*.py`.
Please organize the datasets as follows and run `make_arrow` functions to convert the dataset to pyarrow binary file.

## COCO
https://cocodataset.org/#download

Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)

    root
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...          
    └── karpathy
        └── dataset_coco.json

```python
from vilt.utils.write_coco_karpathy import make_arrow
make_arrow(root, arrows_root)
```

## NLVR2
Clone the [repository](https://github.com/lil-lab/nlvr) and sign the [request form](https://goo.gl/forms/yS29stWnFWzrDBFH3) to download the images.

    root
    ├── images/train           
    │   ├── 0                  
    │   │   ├── train-10108-0-img0.png   
    │   │   └── ...
    │   ├── 1                  
    │   │   ├── train-10056-0-img0.png       
    │   │   └── ...
    │   └── ...
    ├── dev       
    │   ├── dev-0-0-img0.png
    |   └── ...
    ├── test1     
    │   ├── test1-0-0-img0.png
    |   └── ...
    ├── nlvr
    ├── nlvr2
    └── README.md

```python
from vilt.utils.write_nlvr2 import make_arrow
make_arrow(root, arrows_root)
```

## SLAKE

## VQA-RAD
