import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import datetime

import re
import string
import itertools

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
import langid
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder


# nltk.download('punkt') #word_tokenize wont work without

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

#Define function to convert a Unix timestamp to a datetime object
def unix_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)
df["last_played"] = df["last_played"].apply(unix_to_datetime)
df['playtime_at_review']


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
    filtered_tokens = [token for token in tokens if token not in stopwords_list]

    #Lemmitize 
    lemmitized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    #Perform POS tagging
    pos_tagged_tokens = pos_tag(lemmitized_tokens)

    return pos_tagged_tokens

df['preprocessed_text'] = df['review'].apply(lambda x: preprocess_text(x,stopwords, lemmatizer))
df['preprocessed_text']

num_empty = df['preprocessed_text'].apply(lambda x: len(x) == 0).sum()
empty_rows = df[df['preprocessed_text'].apply(lambda x: len(x)) == 0].index

print(f"There are {num_empty} empty rows.")

missing_tags = []

for doc in df['preprocessed_text']:
    for token in doc:
        if len(token) != 2:
            missing_tags.append(token)
            
if len(missing_tags) > 0:
    print("The following tokens are missing a POS tag:", missing_tags)
else:
    print("All tokens have a corresponding POS tag.")

df = df.drop(empty_rows)
df.info()
df['preprocessed_text']


#Word frequency wordcloud & Barchart

freq_dist = FreqDist(df["preprocessed_text"].explode())
fdist = {k[0]:v for k, v in freq_dist.items()}

# Get the top 20 most frequent words from the frequency distribution
top_words = freq_dist.most_common(20)

# Extract the words and their frequencies as separate lists
words = [word[0][0] for word in top_words]
frequencies = [word[1] for word in top_words]

# Bar chart
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
ax1, ax2 = axes.flatten()
ax1.bar(words, frequencies)
ax1.set_title("Frequency Distribution of Preprocessed Steam Reviews", fontsize=16)
ax1.set_xlabel("Words", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12)

# WordCloud object and display it in the second subplot
fdist = {k[0]: v for k, v in freq_dist.items()}
wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(fdist)
ax2.imshow(wordcloud)
ax2.set_title("Word Cloud of Preprocessed Steam Reviews", fontsize=16)
ax2.axis("off")

# Display both plots
plt.tight_layout()
plt.show()
df.info()



# Reviews w.Playtime mean contribution as those users played the game more than average amount
df['playtime_forever'].describe()
playtime_mean = df['playtime_forever'].mean()
df_filtered = df[df['playtime_forever'] >= playtime_mean]
df_filtered.info()

# list of words by concatenating all the preprocessed text in the filtered df
freq_dist_2 = FreqDist(df_filtered['preprocessed_text'].explode())
top_words_2= freq_dist_2.most_common(20)
words_2, freqs = zip(*top_words_2)
words = [w[0] for w in words_2]

# Plot the frequencies
plt.bar(words, freqs)
plt.xticks(rotation=90)
plt.show()



#### Review of users that voted down comment

voted_down = df[df['voted_up']==False]
voted_down_freq = FreqDist(voted_down['preprocessed_text'].explode())
top_voted_down = voted_down_freq.most_common(20)
words_3, freqs_2 = zip(*top_voted_down)
words = [w[0] for w in words_3]
plt.bar(words, freqs_2)
plt.xticks(rotation=90)
plt.show()


### single words do not provide much insight of what features of games are liked vs disliked

### Bi grams with and withour voted_up 

df['cleaned_text'] = df['preprocessed_text'].apply(lambda x: [word[0] for word in x])
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(df['cleaned_text'].sum())
finder.apply_freq_filter(5)  # filter out low-frequency bigrams
bigrams = finder.nbest(bigram_measures.raw_freq, 50)  # get top 50 bigrams by frequency
print(bigrams)

words = [word[0] for i, row in df[df['voted_up']].iterrows() for word in row['preprocessed_text']]
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words)
scored_bigrams = finder.score_ngrams(bigram_measures.raw_freq)
for bigram in scored_bigrams[:10]:
    print(bigram)

### Tri grams with and withour voted_up 

