import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import string
import itertools

import nltk 
from nltk.corpus import sentiwordnet as swn 

from afinn import Afinn
from plotnine import ggplot, aes, geom_bar, labs, xlim

import gensim
from gensim import corpora
from gensim.models import LdaModel

df = pd.read_csv('GTA_V_cleaned.csv')

afinn = Afinn()
afinn.score('love')
afinn.score('toxic')

test = df['cleaned_text'][8]
test_text = ' '.join(test)
score = afinn.score(test_text)
print(score)


def calculate_sentiment_score(text):
    return afinn.score_with_pattern(text)

df['sentiment_score'] = df['cleaned_text'].apply(lambda x: calculate_sentiment_score(' '.join(word[0] for word in x)))
df['sentiment_score'].describe()

df.loc[df['sentiment_score']==-270]

(ggplot(df, aes(x='sentiment_score')) 
 + geom_bar() 
 + labs(x="Sentiment Score", y="Frequency") 
 + xlim(-5, 5)
)
negative_reviews = df[df['sentiment_score'] < 0]
sample_neg_reviews = negative_reviews.sample(n=10, random_state=42)

for index, row in sample_neg_reviews.iterrows():
    print(f"Review: {' '.join(row['cleaned_text'])}")
    print(f"Sentiment score: {row['sentiment_score']}\n")

positive_reviews = df[df['sentiment_score']>0]
sample_pos_reviews = positive_reviews.sample(n=10, random_state=42)
for index, row in sample_pos_reviews.iterrows():
    print(f"Review: {' '.join(row['cleaned_text'])}")
    print(f"Sentiment score: {row['sentiment_score']}\n")
### Overall sentiment using AFINN shows that game rarely gets below a nuetral (0) but also has just as many users with positive sentiment as nuetral

###Need to SentiWordNet include POS_TAG

swn_words = [word[0] for row in df['preprocessed_text'] for word in row]
synsets = swn.senti_synsets('good', 'a')  # 'a' indicates that the word is an adjective
pos_score = neg_score = obj_score = 0

for syn in synsets:
    pos_score += syn.pos_score()
    neg_score += syn.neg_score()
    obj_score += syn.obj_score()

if len(list(synsets)) >0:
    pos_score /= len(synsets)
    neg_score /= len(synsets)
    obj_score /= len(synsets)
print(f"Positive score: {pos_score:.2f}, Negative score: {neg_score:.2f}, Objective score: {obj_score:.2f}") #Retirating what previously seen, overall postive from all collected words
df['preprocessed_text']
preprocessed_docs = [[word for word, tag in doc] for doc in df['preprocessed_text']]
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(text) for text in preprocessed_docs]
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {}\nWords: {}".format(idx, topic))
