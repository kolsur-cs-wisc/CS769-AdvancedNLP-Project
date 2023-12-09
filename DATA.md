# Dataset Preparation
We utilize 4 datsets: COCO Captions (COCO), Natural Language for Visual Reasoning 2 (NLVR2), SLAKE, VQA-RAD

Please download the datasets by yourself.
We use `pyarrow` to serialize the COCO and NLVR2 datasets, conversion scripts are located in `vilt/utils/write_*.py`.
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
## Biomedical Dataset Preparation
For the biomedical datasets, please download the images from the links provided and move them to their respective ```root_{dataset_name}``` folder.

## SLAKE
You can learn about this dataset from this link [SLAKE](https://www.med-vqa.com/slake/#gt-Download). The dataset can be downloaded from the [Google Drive](https://drive.google.com/file/d/1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U/view) folder. 
After downloading the zip file, please extract its contents and copy over the entire ```imgs``` folder to the ```root_slake``` folder. Please note not to replace the json files present in our folder which includes the filtered English only dataset.

    root_slake
    ├── imgs       
    │   ├── xmlab52                 
    │   │   ├── source.jpg   
    │   │   └── ...
    │   └── ...
    ├── test.json       
    ├── train.json     
    └── validate.json

## VQA-RAD
Download the dataset images from the given [link](https://osf.io/89kps/files/osfstorage). Extract the folder zip file. Copy over the entire foler ```osfstorage-archive``` to the ```root_vqarad``` folder in the project. Ensure not to change the provided ```.csv``` files which contain the train-test splits.

    root_vqarad
    ├── osfstorage-archive       
    │   ├── VQA_RAD Image Folder                 
    │   │   ├── synpic47964.jpg   
    │   │   └── ...
    │   └── ...
    ├── medical-vocab.csv       
    ├── test-dataV2.csv
    ├── tokens.json     
    └── train-dataV2.csv