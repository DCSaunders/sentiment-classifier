#!/usr/bin/env python
from __future__ import division
import argparse
import collections
import copy
import cPickle
import numpy as np
import os
import re
import string
import sys
from numpy import log
from scipy.stats import norm

STOPWORDS = set([',', '.', 'the', 'a', 'an', 'of', 'to', 'and', 'is', '"', 'in', "'s", 'that', 'it', ')', '(', 'with', 'I', 'as', 'for', 'film', 'this', 'his', 'her', 'he', 'their', 'they'])

GENERIC_PUNC = re.compile(r"(\w*-?\w*)(--|\.\.\.|[,!?%`./();$&@#:\"'])(\w*-?\w*)") 

POS = 'POS'
NEG = 'NEG'

class Review(object):
    def __init__(self, path):
        self.path = path
        self.rating = 1 if POS in path else -1
        self.text = []
        self.text_no_stopwords = []
        self.bag_ngrams = {1: collections.defaultdict(int),
                           2: collections.defaultdict(int),
                           3: collections.defaultdict(int)}
        self.first_in_sentence = collections.defaultdict(int)
        self.stopwords = 0
        self.topic_words = []
        self.topic_counts = []
        self.topic_probs = []
        
    def lexicon_score(self, lexicon):
        score = 0
        for token in self.bag_ngrams[1]:
            if token in lexicon:
                score += self.bag_ngrams[1][token] * lexicon[token]
        if (self.rating * score > 0):
            return 1
        else:
            return 0

    def train_ngrams(self, freqs, to_recase, recase=False, ngram=1):
        for tok, freq in self.bag_ngrams[ngram].items():
            freqs[tok] += freq
        if recase:
            for tok, freq in self.first_in_sentence.items():
                to_recase[tok] += freq
    
    def tokenize(self, vocab):
        with open(self.path, 'r') as f:
            for line in f:
                for index, word in enumerate(line.split()):
                    split_word = space_punctuation(word)
                    self.text.extend(split_word)
                    if (index == 0):
                        self.first_in_sentence[split_word[0]] += 1
                    for seg in split_word:
                        self.bag_ngrams[1][seg] += 1
                        if index == 0:
                            seg = seg.lower()
                        if seg in STOPWORDS:
                            self.stopwords += 1
                        else:
                            self.text_no_stopwords.append(seg)
                            vocab.add(seg)
        self.get_ngrams()
        
    def get_ngrams(self):
        bigrams = zip(self.text, self.text[1:])
        for token in bigrams:
            self.bag_ngrams[2][token] += 1
        trigrams = zip(self.text, self.text[1:], self.text[2:])
        for token in trigrams:
            self.bag_ngrams[3][token] += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    args = parser.parse_args()
    return args

def space_punctuation(word):
    matches = []
    m = re.findall(GENERIC_PUNC, word)
    for match in m:
        if "'" in match:
            match = adjust_apostrophe(match)
        for seg in match:
            if seg:
                matches.append(seg)
    if not matches:
        matches = [word]
    return matches

def adjust_apostrophe(match_tuple):
    if match_tuple[2] == 't':
        return tuple([match_tuple[0][:-1], "n't"])
    else:
        return tuple([match_tuple[0], "'{}".format(match_tuple[2])])

def walk_dir(path_to_dir):
    path_list = []
    for dirpath, dirnames, filenames in os.walk(path_to_dir):
        for filename in filenames:
            path_list.append(os.path.join(dirpath, filename))
    return path_list

def tokenize_files(doc_dir, docs, vocab):
    doc_paths = walk_dir(doc_dir)
    for path in doc_paths:
        new_doc = Review(path)
        new_doc.tokenize(vocab)
        docs.append(new_doc)
