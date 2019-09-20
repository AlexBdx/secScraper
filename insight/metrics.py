#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[4]:


from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import difflib

import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import re
import Load_MasterDictionary as LM
import string

# # Metrics

# ## Jaccard similarities

def sim_jaccard(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def test_sim_jaccard():
    Da = "We expect demand to increase."
    Db = "We expect worldwide demand to increase."
    Dc = "We expect weakness in sales."
    test1 = sim_jaccard(Da, Db)
    test2 = sim_jaccard(Da, Dc)
    assert round(test1, 2) == 0.83
    assert round(test2, 2) == 0.25
    return True

# test_sim_jaccard()


# ## TF & TF-IDF similarities (cosine similarities)

# In[9]:


# Cosine similarity
def sim_cosine(str1, str2):
    vect = TfidfVectorizer(use_idf=False)  # Per paper
    tfidf = vect.fit_transform([str1, str2])
    tfidf_similarity = tfidf * tfidf.T
    return float(tfidf_similarity[0, 1])

def test_sim_cosine():
    Da = "We expect demand to increase."
    Db = "We expect worldwide demand to increase."
    Dc = "We expect weakness in sales."
    test1 = sim_cosine(Da, Db)
    test2 = sim_cosine(Da, Dc)
    assert round(test1, 2) == 0.91
    assert round(test2, 2) == 0.40
    return True

# test_sim_cosine()


# ## Sequence modifications

# In[11]:


def sim_minEdit(str1, str2):
    """
    This is character based.
    WARNING: VERY SLOW BEYOND ~10,000 CHAR TO COMPARE"""
    f = difflib.SequenceMatcher(None, a=str1, b=str2)
    count_words_str1 = len(re.compile(r'\w+').findall(str1))
    count_words_str2 = len(re.compile(r'\w+').findall(str2))
    transformations = f.get_opcodes()  # Impossible to compute for larger texts
    transformations = [t for t in transformations if t[0] != 'equal']
    similarity = 1-len(transformations)/(count_words_str1+count_words_str1)
    # similarity = f.ratio()
    return similarity

def test_sim_minEdit():
    Da = "We expect demand to increase."
    Db = "We expect worldwide demand to increase."
    Dc = "We expect weakness in sales."
    test1 = sim_minEdit(Da, Db)
    test2 = sim_minEdit(Da, Dc)
    assert round(test1, 2) == 0.90
    assert round(test2, 2) == 0.30
    return True

# test_sim_minEdit()


# In[26]:


def sim_simple(str1, str2):
    """This is word based
    WARNING: VERY SLOW BEYOND ~10,000 CHAR TO COMPARE"""
    d = difflib.Differ()
    count_words_str1 = len(re.compile(r'\w+').findall(str1))
    count_words_str2 = len(re.compile(r'\w+').findall(str2))
    comparison = list(d.compare(str1, str2))
    comparison = [change for change in comparison if change[0] != ' ']
    similarity = 1-len(comparison)/(len(str1) + len(str2))
    return similarity

def test_sim_simple():
    Da = "We expect demand to increase."
    Db = "We expect worldwide demand to increase."
    Dc = "We expect weakness in sales."
    test1 = sim_simple(Da, Db)
    test2 = sim_simple(Da, Dc)
    assert round(test1, 2) == 0.85
    assert round(test2, 2) == 0.67
    return True

# test_sim_simple()

"""Sentiment analysis"""
MASTER_DICTIONARY_FILE = r'LoughranMcDonald_MasterDictionary_2018.csv'
OUTPUT_FIELDS = ['file type', 'file size', 'number of words', '% positive', '% negative',
                 '% uncertainty', '% litigious', '% modal-weak', '% modal moderate',
                 '% modal strong', '% constraining', '# of alphabetic', '# of digits',
                 '# of numbers', 'avg # of syllables per word', 'average word length', 'vocabulary']
lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)
def sim_sentiment(text):
    """Create a composite index based on the sentiment output"""
    # Warning: Values are in absolute value
    data = sentiment(text)
    #print(data)
    # Sign will be of positive + negative proportion. Will be dialed down based on neutrals
    if (data[3]+data[4]+data[5]+data[6]):
        composite_index = (data[3]-data[4]-data[6])/(data[3]+data[4]+data[5]+data[6])
    else:  # Avoid the case when the text is too short and a div per zero error is thrown
        composite_index = 0
    
    return composite_index


def sentiment(text):
    text_len = len(text)
    text = re.sub('(May|MAY)', ' ', text)  # drop all May month references ## lol
    text = text.upper()  # for this parse caps aren't informative so shift
    output_data = get_data(text)
    output_data[0] = type(text)
    output_data[1] = text_len
    return output_data

# Helper function - should not be accessed from the outside
def get_data(doc):
    vdictionary = {}
    _odata = [0] * 17
    total_syllables = 0
    word_length = 0
    
    tokens = re.findall('\w+', doc)  # Note that \w+ splits hyphenated words
    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[2] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1
            if lm_dictionary[token].positive: _odata[3] += 1
            if lm_dictionary[token].negative: _odata[4] += 1
            if lm_dictionary[token].uncertainty: _odata[5] += 1
            if lm_dictionary[token].litigious: _odata[6] += 1
            if lm_dictionary[token].weak_modal: _odata[7] += 1
            if lm_dictionary[token].moderate_modal: _odata[8] += 1
            if lm_dictionary[token].strong_modal: _odata[9] += 1
            if lm_dictionary[token].constraining: _odata[10] += 1
            total_syllables += lm_dictionary[token].syllables

    _odata[11] = len(re.findall('[A-Z]', doc))
    _odata[12] = len(re.findall('[0-9]', doc))
    # drop punctuation within numbers for number count
    doc = re.sub('(?!=[0-9])(\.|,)(?=[0-9])', '', doc)
    doc = doc.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    _odata[13] = len(re.findall(r'\b[-+\(]?[$€£]?[-+(]?\d+\)?\b', doc))
    _odata[14] = total_syllables / _odata[2]
    _odata[15] = word_length / _odata[2]
    _odata[16] = len(vdictionary)
    
    # Convert counts to %
    for i in range(3, 10 + 1):
        _odata[i] = (_odata[i] / _odata[2]) * 100
    # Vocabulary
        
    return _odata
    




