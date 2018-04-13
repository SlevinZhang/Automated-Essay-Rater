# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:38:05 2018

@author: siliangzhang
"""
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize,RegexpTokenizer
import re

def pos_tagging_feature(essay):
    """
        we only need 
        'NN/NNS/NNP/NNPS' -> noun
        'VB/VBD/VBG/VBN/VBP/VBZ' -> verb
        'JJ/JJR/JJS' -> adjective
    """
    word_tokens = word_tokenize(essay)
    words_tagged = nltk.pos_tag(word_tokens)
    
    tag_freq = nltk.FreqDist([tag for (word, tag) in words_tagged])
    
    tag_list = tag_freq.most_common()
    print(tag_list)
    tag_dict = {}
    tag_dict['NN'] = sum([ pair[1] for pair in tag_list if re.match(r"NN.*",pair[0])])
    tag_dict['VB'] = sum([pair[1] for pair in tag_list if re.match(r"VB.*",pair[0])])
    tag_dict['JJ'] = sum([pair[1] for pair in tag_list if re.match(r"JJ.*",pair[0])])
    
    return tag_dict

def statistical_feature(essay):
    
    #average sentence length
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sent_list = sent_detector.tokenize(essay.strip())
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    #sentence count
    num_sentence = len(sent_list)

    #average length
    average_sent_length = sum([len(tokenizer.tokenize(sentence)) for sentence in sent_list ]) / num_sentence
    
    
    word_tokens = tokenizer.tokenize(essay)
    
    #lower case
    word_tokens = [w.lower() for w in word_tokens]
    
    #remove stopwords

    stop_words = set(stopwords.words('english'))
    
    word_tokens = [w for w in word_tokens if w not in stop_words]
    
    #words count
    num_words = len(word_tokens)
    
    statis_dict = {}
    statis_dict['num_words'] = num_words
    statis_dict['num_sentences'] = num_sentence
    statis_dict['average_sent_length'] = average_sent_length

    return statis_dict