import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import string
import itertools

import nltk 
from nltk.corpus import sentiwordnet as swn 
nltk.download('omw-1.4')

from afinn import Afinn
from plotnine import ggplot, aes, geom_bar, labs, xlim

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel


df = pd.read_csv('GTA_V_cleaned.csv')
df.head()

afinn = Afinn()
afinn.score('love')
afinn.score('toxic')

test = df['cleaned_text'][10]
test_text = ' '.join(test)
score = afinn.score(test)
print(score)


def calculate_sentiment_score(text):
    return afinn.score(text)

df['sentiment_score'] = df['cleaned_text'].apply(calculate_sentiment_score)
df['sentiment_score'].describe()

lower_outl = df.loc[df['sentiment_score']==-270]
lower_outl['review']

(ggplot(df, aes(x='sentiment_score')) 
 + geom_bar() 
 + labs(x="Sentiment Score", y="Frequency") 
 + xlim(-5, 5)
)
negative_reviews = df[df['sentiment_score'] < 0]
sample_neg_reviews = negative_reviews.sample(n=10, random_state=42)
sample_neg_reviews


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
df['cleaned_text']


preprocessed_docs = [doc.split() for doc in df['cleaned_text']]
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(text) for text in preprocessed_docs]
texts = [eval(text) for text in df['cleaned_text']]
coherence_dict = gensim.corpora.Dictionary(texts)
num_topics = 8
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics,
                                            random_state=42,
                                            passes=10,
                                            per_word_topics=True)

for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx}')
    keywords = ", ".join(word for word, _ in lda_model.show_topic(idx, topn=10))
    print(f'Top Keywords: {keywords}\n')



coherence_model = CoherenceModel(
    model=lda_model,
    texts=texts,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()
print(f"Coherence Score: {coherence_score}")
# def get_topic_name(model, idx, num_words = 3):