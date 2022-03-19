## Introduction of Plane Geometry Diagram Parsing (PGDP)

The code and dataset for IJCAI 2022 Paper "*Plane Geometry Diagram Parsing*" [PDF].

We propose the **PGDPNet**, the first end-to-end deep learning model for explicit geometry diagram parsing. And we construct a large-scale dataset **PGDP5K**, containing dense and fine-grained annotations of primitives and relations. Our method demonstrates superior performance of diagram parsing, outperforming previous methods remarkably.
<div align=center>
	<img src="framework.png">
	
</div>
<div align=center>
	Figure 1. Framework of PGDPNet
</div>

<div align=center>
	<img src="compare.png">
</div>
<div align=center>
	Figure 2. Compare with SGG
</div>

## PGDP5K Dataset
You could download the dataset from:
- [[BaiduYun link](https://pan.baidu.com/s/1RVArHmqmaA-P7ba53ue75Q)], _keyword_: a5sj
- [[GoogleDrive link](https://drive.google.com/file/d/1UnGJO70Tth8u_PAu3UiHixevw-_UfxrP/view?usp=sharing)]

## Environmental Settings
- python version: **3.8.3**
- CUDA version: **10.1**
- Other settings please refer to *requirements.txt*

We use **4 NVIDIA 1080ti GPUs** for the training and more GPUs with large batch size will bring a little performance improvment.

## Pretrain models

We provide the pretrain backbone of [resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) or [resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) trained on ImageNet , which will put in the fold of **./pretrain_models**.
And we also provide the best model of resnest50 [BaiduYun link](https://pan.baidu.com/s/1stfwhTeEALCofVFIKqGmvw), keyword: g2xy, [GoogleDrive link](https://drive.google.com/file/d/1VwzrvMU7M5gux5ZPyZPgMK4PizT3z7qi/view?usp=sharing)


## Usage

Decompress three json files of train/val/test datasets in the fold of **./dataset/AiProducts**, more details please see the config files in the fold of **./configs**.

```bash
# Get dataset of AiProducts
sh ./dataset/get_dataset_AiProducts.sh
```

```bash
# Training the model and the models will be saved in the fold of ./log/AiProducts 
python main.py  
```

```bash
# Finetune the model
python main.py --RESUME_MODEL ./log/AiProducts/best_model.pth --DATASET_TRAIN_JSON ./dataset/AiProducts/converted_val.json 
```
 
```bash
# Test the model
python test.py --RESUME_MODEL ./log/AiProducts/best_model.pth
```

```bash
# Adjust the classifier using the tau-norm method and the models will be saved in the fold of ./log_tau
python tau_norm.py --RESUME_MODEL ./log/AiProducts/best_model.pth
```

More experiments need to be tried such as different *image size*, *backbone*, *optimizer* or *learning rate decay method* which noly need change the config file.

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with `fpath` and `category_id`.

Here is an example.
```
AiProducts
{
    'annotations': 
	[
        {
		"category_id": 0, 
		"fpath": "/val/00000/1849756.jpg"
        },
        ...
    ]
    'num_classes': 50030
}
```

## Citation

If the paper, the dataset, or the code helps you, please cite the paper in the following format:


## Acknowledge

Please let me know if you encounter any issues. You could contact with the first author (zhangmingliang2018@ia.ac.cn) or leave an issue in the github repo.
