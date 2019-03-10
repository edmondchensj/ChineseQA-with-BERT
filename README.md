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
We format the DuReader dataset in the format identical to what BERT uses for the SQuAD dataset:
```

```

## BiDirectional Encoder Representations (BERT)

[BERT](https://github.com/google-research/bert) is a new method of pre-training transformers for a variety of NLP tasks, including QA-IR. It achieved state of the art results on the SQuAD dataset so we wanted to apply it to DuReader. 

### Train BERT-Chinese on the Dataset
First install the PyTorch implementation of BERT with Huggingface's python package (https://github.com/huggingface/pytorch-pretrained-BERT):
```
pip install pytorch-pretrained-bert
```
Using our preprocessed training files or larger training sets obtained with our preprocessing script, run the training script with the command:
```
python run_dureader.py --bert_model bert-base-chinese --do_train --do_lower_case --train_file data/20000_search.train.json   --train_batch_size 12 --gradient_accumulation_steps 3   --learning_rate 3e-5   --num_train_epochs 2.0   --max_seq_length 384   --doc_stride 128   --output_dir ../duoutput
```
The hyperparameter setting has been tested on GTX1080 8GB. Generating features from the training file for the Chinese IR task can take a long time with the current scripts and it is CPU-only. For a training set of size 10000, it takes about 8-10 hours on our setup. Training itself is 1-2 hours depending on the hyperparameters.
After training is complete, use the following command to generate predictions for the 1000-example preprocessed development set included in this repository:
```
python run_dureader.py --bert_model ../duoutput --do_predict --predict_file data/1000_search.dev.json --max_seq_length 384 --doc_stride 128 --output_dir ../duprediction
```
## BLEU Scoring

BLEU scoring is an algorithm used to evaluate the quality of text. It has a fairly high correlation to human judgement and is significantly better than accuracy. DuReader evaluates results based off of the BLEU scoring metric.

### Get BLEU scoring

The BERT model will output `predictions.json`. 

```
python3 bert_bleu.py [path/to/predictions.json] [path/to/preprocessed.json]
```
