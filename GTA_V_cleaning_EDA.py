import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns


import re 
import string
import itertools

import nltk
from nltk import stopwords
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
nltk.download('punkt') #word_tokenize wont work without

df = pd.read_csv('GTA_V.csv')
df.info()
df['received_for_free'].value_counts(normalize=True)
df['voted_up'].value_counts(normalize=True)
df['weighted_vote_score'].value_counts(normalize=True) # Weighted vote score severly imbalanced with 88%+ in 0

df['author'][0]


#Want key, value pair within Author column to be its own columns, so need to extract 

def extract_author_data(author_string):
    author_dict = eval(author_string)  # convert the string to a dictionary
    return author_dict

df['author_data'] = df['author'].apply(extract_author_data)
df = pd.concat([df.drop(['author_data'], axis=1), df['author_data'].apply(pd.Series)], axis=1)
df = df.drop(['author'], axis=1)

df.drop(['steam_china_location'], axis=1, inplace=True)
df.dropna(inplace=True)
df.info()