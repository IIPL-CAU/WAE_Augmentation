import os
import re
import random
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def read_data(data_name: str, data_path: str):
    """
    Remove html tags and special characters from given text

    Args:
        data_name (str): given dataset name
        data_path (str): saved dataset path
    Returns:
        train_dat, test_dat (pd.DataFrame): loaded dataframe
    """
    if data_name == 'AG_News':
        train_dat = pd.read_csv(os.path.join(data_path , 'AG_News/train.csv'), sep=',',
                                names=['label', 'title', 'description'])
        test_dat = pd.read_csv(os.path.join(data_path , 'AG_News/test.csv'), sep=',',
                                names=['label', 'title', 'description'])

        train_dat['total_text'] = train_dat['title'] + '[SEP]' + train_dat['description']
        test_dat['total_text'] = test_dat['title'] + '[SEP]' + test_dat['description']
        

    if data_name == 'DBPia':
        train_dat = pd.read_csv(os.path.join(data_path , 'DBPia/train.csv'), sep=',',
                                names=['label', 'title', 'description'])
        test_dat = pd.read_csv(os.path.join(data_path , 'DBPia/test.csv'), sep=',',
                                names=['label', 'title', 'description'])

        train_dat['total_text'] = train_dat['title'] + '[SEP]' + train_dat['description']
        test_dat['total_text'] = test_dat['title'] + '[SEP]' + test_dat['description']

    if data_name == 'IMDB':
        train_dat = pd.read_csv(os.path.join(data_path , 'IMDB/train.csv'), sep='\t',
                                names=['label', 'description'])
        test_dat = pd.read_csv(os.path.join(data_path , 'IMDB/TEST.csv'), sep='\t',
                                names=['label', 'description'])

        train_dat['total_text'] = train_dat['description']
        test_dat['total_text'] = test_dat['description']

    if data_name == 'Yelp_Full':
        train_dat = pd.read_csv(os.path.join(data_path , 'Yelp_Full/train.csv'), sep=',',
                                names=['label', 'description'])
        test_dat = pd.read_csv(os.path.join(data_path , 'Yelp_Full/TEST.csv'), sep=',',
                                names=['label', 'description'])

        train_dat['total_text'] = train_dat['description']
        test_dat['total_text'] = test_dat['description']

    return train_dat, test_dat

def train_valid_split(dataframe_: pd.DataFrame, valid_split_ratio: float):
    total_length = len(dataframe_)
    split_count = int(len(dataframe_) * valid_split_ratio)

    valid_index = np.random.choice(total_length, split_count, replace=False)
    train_index = list(set(range(total_length)) - set(valid_index))

    train_dat = dataframe_.iloc[train_index]
    valid_dat = dataframe_.iloc[valid_index]

    return train_dat, valid_dat

def clean_text(text: str):
    """
    Remove html tags and special characters from given text

    Args:
        text (str): given original sentence
    Returns:
        cleantext (str): cleaned sentence
    """
    cleantext = BeautifulSoup(text, "lxml").text

    cleantext = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE) # Remove URL
    cleantext = re.sub('[^A-Za-z.,?!\']+', ' ', cleantext)

    cleantext = re.sub(r"\.+",". ", cleantext)
    cleantext = re.sub(r"\,+",", ", cleantext)
    cleantext = re.sub(r"\?+","? ", cleantext)
    cleantext = re.sub(r"\!+","! ", cleantext)
    cleantext = re.sub("\s+"," ", cleantext)

    return cleantext