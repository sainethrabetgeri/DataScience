#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 08:53:47 2023

@author: nethrachekuri
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

text = """ A week ago a friend invited a couple of other couples over for dinner. 
Eventually, the food (but not the wine) was cleared off the table for what turned out to be some fierce Scrabbling. Heeding the strategy of going for the shorter, more valuable word over the longer cheaper word, our final play was “Bon,” which–as luck would have it!–happens to be a Japanese Buddhist festival, and not, as I had originally asserted while laying the tiles on the board, one half of a chocolate-covered cherry treat. Anyway, the strategy worked. My team only lost by 53 points instead of 58.
Just the day before, our host had written of the challenges of writing short"""

stopwords = list(STOP_WORDS)
len(stopwords)

nlp = spacy.load('en_core_web_s m')

doc = nlp(text)
 
tokens = [token.text for token in doc]
print(tokens)

word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                 word_frequencies[word.text] = 1
            else:
                  word_frequencies[word.text] += 1
            

max_frequency = max(word_frequencies.values())
max_frequency

for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency

sentence_tokens =  [sent for sent in doc.sents]
len(sentence_tokens)

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] = word_frequencies[word.text.lower()]

from heapq import nlargest
select_length = int(len(sentence_tokens)*0.5)
select_length

summary = nlargest(select_length,sentence_scores, key = sentence_scores.get)
final_summary = [word.text for word in summary]


            
      

