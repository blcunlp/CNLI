# Baseline for Chinese Natural Language Inference (CNLI)  dataset

## Description
This repository provides the official training and development dataset for [the Chinese Natural Language Inference (CNLI) share task] (http://www.cips-cl.org/static/CCL2018/call-evaluation.html). 

We evaluate the cnli\_1.0 corpus on two baseline models. 


## Data

The CNLI dataset can be downloaded at [here](https://github.com/blcunlp/CNLI/tree/master/CNLI_Data)

Both the train and dev set are  **tab-separated** format.
Each line in the train (or dev) file corresponds to an instance, and it is arranged as：  
>sentence-id premise   hypothesis  label



## Model

This repository includes the baseline model for Chinese Natural Language Inference (CNLI) dataset. 
We provide two baseline models. 
(1) The [Decomposable Attention Model](https://arxiv.org/pdf/1606.01933.pdf), which use FNNs and inter-attention mechinaism. More details about the model can be found in the [original paper](https://arxiv.org/pdf/1606.01933.pdf). 
(2) The ESIM Model (https://arxiv.org/pdf/1609.06038.pdf), which is a strong baseline model for SNLI dataset. 

## Requirements
* python 3.5
* tensorflow      '1.4.0'
* jieba 0.39

## Training


**Data Preprocessing**  
We use jieba to tokenize the sentences. During trainging, we use the pre-trained SGNS embedding introduced in [Analogical Reasoning on Chinese Morphological and Semantic Relations] (https://arxiv.org/abs/1805.06504).  You can download the sgns.merge.word from [here](https://pan.baidu.com/s/1kwxiPouou6ecxyJdYmnkvw).

**Main Scripts**  
config.py：the parameter configuration.  
decomposable_att.py: implementation of the Decomposable Attention Model.   
data_reader.py: preparing data for the model.    
train.py: training the Decomposable Attention Model. 

**Running Model**  
You can train the decomposable attention model and the esim model by the following command lines: 
> python3 train.py  --model_type  decomposable_att 
> python3 train.py  --model_type  esim



## Results 
We provide the whole training data, which comprimises 90,000 items in the training set and 10,000 items in the dev dataset. 
We adopt early stopping on dev set. The best results are shown in the following table: 

|Model |train-acc(%)|dev-acc(%)
|:-:|:-:|:-:
| Decomposable-Att|76.91 |69.35
|ESIM |  76.82| 73.57



## Reporting issues
Please let us know, if you encounter any problems.
