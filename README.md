# Text Augmentation with WAE (Wasserstein Auto-Encoder)

Official implementation of "Generative Data Augmentation via Wasserstein Autoencoder for Text Classification". (ICTC 2022)

https://ieeexplore.ieee.org/document/9952762/

### Dependencies

This code is written in Python. Dependencies include

* Python == 3.6
* PyTorch == 1.8
* Transformers (Huggingface) == 4.8.1

### Usable Data
* **IMDB** [Sentiment Analysis] (--dataset=IMDB)
* **Yelp Open** [Reveiw Classification] (--dataset=Yelp_Full)
* **DBpedia** [Query Relationship Analysis] (--dataset=DBpedia)
* **AG News** [Topic Classification] (--dataset=AG_News)
* **SST2** [Sentiment Analysis] (--dataset=SST2)
* **SST5** [Sentiment Analysis] (--dataset=SST5)
* **ProsCons** [Sentiment Analysis] (--dataset=ProsCons)
* **Subjectivity** [Sentiment Analysis] (--dataset=SUBJ)
* **Text REtrieval Conference** [Question Classification] (--dataset=TREC)
* **Movie Review** [Sentiment Analysis] (--dataset=MR)

### Pre-processing

```
python3 main.py --preprocessing --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**
- preprocessed_path: reprocessed data file save path
- valid_split_ratio: Validation dataset split ratio; Default is 0.05
- sentencepiece_model: Google's SentencePiece model type; Default is unigram
- pad_idx: Padding token index; Default is 0
- unk_idx: Unknown token index; Default is 3
- bos_idx: Start token index; Default is 1
- eos_idx: End token index; Default is 2
- min_len: Minimum length of sequence; Default is 4
- max_len: Maximum length of sequence; Default is 300

**After execution, These files will be generated**
- processed pickle file

### Augmentation Training

```
python3 main.py --augment_training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**
- ae_type: Auto-encoder type; Default is WAE
- aug_tokenizer: Tokenizer settings; Default is T5
- aug_model_type: Augmentation model settings; Default is T5
- WAE_loss: Wasserstein Auto-encoder Loss Type; Default is mmd
- d_latent: Latent space dimension; Default is 256


**After execution, These files will be generated**
- Augmentation trained model weights

### Augmentation

```
python3 main.py --augmentation --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv
- model weights

**After execution, These files will be generated**
- Augmented data

### Classification Training

```
python3 main.py --training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv
- augmented data (Optional)

**Additional arguments**
- train_with_augmentation: Text classifier training with augmentation data; Default is True
- train_only_augmentation: Text classifier training only with augmentation data; Default is False
- cls_tokenizer: Classification tokenizer settings; Default is T5
- cls_model_type: Classification model settings; Default is T5
