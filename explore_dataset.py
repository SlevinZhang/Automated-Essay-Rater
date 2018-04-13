# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:13:00 2018

@author: siliangzhang
"""
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
dataset_root = './dataset/'
filename = 'globalenglish_essay_scoring.csv'

df = pd.read_csv(dataset_root + filename,encoding='latin-1')
print("=========================================")
print(df.head(5))

print("total number of essay: {}".format(len(df.index)))

sample_essay = ''
for index in range(df.shape[0]):
    sample_essay += df['essay'][index]
print("=========================================")
print(sample_essay[:20])

custom_stop_words = set(["''","``","'s","n't"])

stop_words = set(stopwords.words('english')) | custom_stop_words

word_tokens = word_tokenize(sample_essay)

#remove punctuation
word_tokens = [w for w in word_tokens if len(w) > 1]

#lowercase all words
word_tokens = [w.lower() for w in word_tokens]

filtered_words = [w for w in word_tokens if w not in stop_words]

print(filtered_words[:20])
print(len(filtered_words))

#count the frequency of words
fdist = nltk.FreqDist(filtered_words)

for word, freq in fdist.most_common(100):
    print('{}:{}'.format(word,freq))