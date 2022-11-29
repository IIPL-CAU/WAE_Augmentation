# Text Style Transfer & Augmentation with WAE (Wasserstein Auto-Encoder)

Official implementation of "Generative Data Augmentation via Wasserstein Autoencoder for Text Classification". (ICTC 2022)

https://ieeexplore.ieee.org/document/9952762/

### Dependencies

This code is written in Python. Dependencies include

* Python == 3.6
* PyTorch == 1.8
* Transformers (Huggingface) == 4.8.1

### Usable Data
['IMDB', 'Yelp_Full', 'DBpedia', 'AG_News', 'SST2', 'SST5', 'ProsCons', 'SUBJ', 'TREC', 'MR']
* IMDB **Sentiment Analysis** (--dataset=IMDB)
* Yelp Open **Reveiw Classification** (--dataset=Yelp_Full)
* DBpedia **Query Relationship Analysis** (--dataset=DBpedia)
* AG News **Topic Classification** (--dataset=AG_News)
* SST2 **Sentiment Analysis** (--dataset=SST2)
* SST5 **Sentiment Analysis** (--dataset=SST5)
* ProsCons **Sentiment Analysis** (--dataset=ProsCons)
* Subjectivity **Sentiment Analysis** (--dataset=SUBJ)
* Text REtrieval Conference **Question Classification** (--dataset=TREC)
* Movie Review **Sentiment Analysis** (--dataset=MR)

### Pre-processing

```
python3 main.py --preprocessing --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**


### Augmentation Training

```
python3 main.py --augment_training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**

### Augmentation

```
python3 main.py --augmentation --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**

### Classification Training

```
python3 main.py --training --dataset=IMDB
```

**These arguments must be provided**
- dataset: Name of a directory below dataset_path, which contations train.csv and test.csv

**Additional arguments**


**After execution, These files will be generated**