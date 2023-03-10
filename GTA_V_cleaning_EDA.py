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
# nltk.download('punkt') #word_tokenize wont work without

df = pd.read_csv('GTA_V.csv')
df.info()
test_review = df['review'].loc[200]
test_review