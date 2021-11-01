# K-BERT

Paddle implementation of ["K-BERT: Enabling Language Representation with Knowledge Graph"](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuW.5594.pdf).


## Requirements

Software:
```
Python3.7

PaddlePaddle2.1.2
```
## Datasets

XNLI

## Prepare

* Download the ``paddle_weight.pdparams`` from [here](https://aistudio.baidu.com/aistudio/datasetdetail/114344), and save it to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://share.weiyun.com/5BvtHyO), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://share.weiyun.com/5Id9PVZ), unzip and place them in the ``datasets/`` directory.

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
│
├── models
│   ├── google_config.json
│   ├── paddle_weight.pdparams
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
└── run_kbert_cls.py
```


## K-BERT for text classification

### Classification example

Run example on Book review with CnDbpedia:
```sh
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/paddle_weight.pdparams \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/xnli/train.tsv \
    --dev_path ./datasets/xnli/dev.tsv \
    --test_path ./datasets/xnli/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_bookreview_CnDbpedia.pdparams \
    > ./outputs/kbert_bookreview_CnDbpedia.log &
```

Results:
```
Best accuracy in dev : 88.80%
Best accuracy in test: 87.69%
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

### Result

Accuracy (dev/test %) on different dataset:

| Dataset       | Pytorch      | Paddle        |
| :-----        | :----:       | :----:        |
| XNLI          | 77.11/77.07  |               |


