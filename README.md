# EECS 496: Advanced Topics in Deep Learning <br/> Group 7 Final Project: QA IR in Chinese with BERT

## Dataset
The [DuReader dataset](https://github.com/baidu/DuReader) is a machine reading comprehension dataset in Chinese. It is the rough equivalent to the popular [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuAD) in English. 

### Download the Dataset
To Download DuReader dataset:
```
git clone git@github.com:baidu/DuReader.git
cd DuReader/data && bash download.sh
```

### Preprocess the Dataset

## BiDirectional Encoder Representations (BERT)

[BERT](https://github.com/google-research/bert) is a new method of pre-training transformers for a variety of NLP tasks, including QA-IR. It achieved state of the art results on the SQuAD dataset so we wanted to apply it to DuReader. 

### Run BERT on the Dataset


## BLEU Scoring

BLEU scoring is an algorithm used to evaluate the quality of text. It has a fairly high correlation to human judgement and is significantly better than accuracy. DuReader evaluates results based off of the BLEU scoring metric.

### Get BLEU scoring

The BERT model will output `predictions.json`. 
