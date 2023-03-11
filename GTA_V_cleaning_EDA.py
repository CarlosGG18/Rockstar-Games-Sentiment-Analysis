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
import langid
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

df['num_games_owned'].describe()
games_owned_outlier = df.loc[df['num_games_owned']== 16506.000000] # despite this record being an outlier for owned the review is real and needs to be kept 
print(games_owned_outlier.loc[22565, 'review'])

#Comment Count 
df['comment_count'].value_counts(normalize=True) # Wont be needing this due to huge imbalance

df['playtime_forever'].value_counts(normalize=True)

df['playtime_forever'].plot() 
plt.show()


### PreProcessing

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, stopwords_list, lemmatizer):
    # If language is not english or has confidence below 0.5 returns empty list
    lang, confidence = langid.classify(text)
    if lang != 'en' or confidence < 0.5:
        return []
    
    #Lowercase text
    text = text.lower()

    #Removing numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)
    
    #Remove stopwords
    filtered_tokens = [token for token in tokens if not in stopwords_list]

    #Lemmitize 
    lemmitized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return lemmitized_tokens

preprocessed_df = df['review'].apply(lambda x: preprocess_text(x,stopwords, lemmatizer))
