# K-BERT

## 1 Introduction

This project reproduces K-BERT based on paddlepaddle framework. Too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue. To overcome KN, K-BERT introduces soft-position and visible matrix to limit the impact of knowledge. K-BERT can easily inject domain knowledge into the models by equipped with a KG without pre-training by-self because it is capable of loading model parameters from the pre-trained BERT.

### Paper:

[<li>K-BERT: Enabling Language Representation with Knowledge Graph</li>](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuW.5594.pdf)

### Reference project：

<li>https://github.com/autoliuweijie/K-BERT</li>


## 2 Accuracy

Accuracy (dev / test %) on XNLI dataset:

| Dataset       | Pytorch        | Paddle         |
| :-----        | :----:         | :----:         |
| XNLI          | 77.11 / 77.07  | 77.55 / 76.75  |

## 3 Environment

<li>Hardware: GPU, CPU</li>

<li>Framework: PaddlePaddle >= 2.1.0</li>

## 4 Datasets

### XNLI

The Cross-lingual Natural Language Inference (XNLI) corpus is the extension of the Multi-Genre NLI (MultiNLI) corpus to 15 languages. The dataset was created by manually translating the validation and test sets of MultiNLI into each of those 15 languages. The English training set was machine translated for all languages. The dataset is composed of 122k train, 2490 validation and 5010 test examples.

## 5 Quick start

### step1 Prepare

* Download the ``paddle_weight.pdparams`` from [here](https://aistudio.baidu.com/aistudio/datasetdetail/114344), and save it to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://share.weiyun.com/5BvtHyO), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://share.weiyun.com/5Id9PVZ), unzip and place them in the ``datasets/`` directory.

### step2 train

Run example on XNLI:
```sh
python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/paddle_weight.pdparams \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/xnli/train.tsv \
    --dev_path ./datasets/xnli/dev.tsv \
    --test_path ./datasets/xnli/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_XNLI.pdparams
```

Options of ``run_kbert_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
        [--output_model_path] - Path to the output model.
```
## 6 Code structure

The directory tree of K-BERT:
```
K-BERT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── datasets
│   ├── xnli
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│    ...
├── models
│   ├── google_config.json
│   ├── paddle_weight.pdparams
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
└── run_kbert_cls.py
```
