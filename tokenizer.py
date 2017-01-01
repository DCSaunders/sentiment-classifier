#!/usr/bin/env python
from __future__ import division
import argparse
import collections
import copy
import cPickle
import logging
import numpy as np
import os
import re
import string
import sys
from numpy import log
from scipy.stats import norm

STOPWORDS = set([',', 'the', '.', 'a', 'and', 'of', 'to', 'is', 'in', "'s", '"', "'", '`', ';', '--', 'that', 'it', 'as', 'his', ')', '(', 'with', 'he', 'for', 'film', 'this', 'but', 'I', 'on', 'an', 'are', 'by', 'who', 'not', 'has', 'be', 'from', "n't", 'you', 'at', 'one', 'was', 'have', 'movie', 'all', 'they', 'which', 'like', 'him', 'do', 'more', 'about', 'so', 'there', 'her', 'when', 'we', 'out', 'up', 'does', 'she', 'their', 'its'])

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
                           2: collections.defaultdict(int)}
        self.first_in_sentence = collections.defaultdict(int)
        self.stopwords = 0
        self.topic_words = []
        self.topic_counts = []
        
    def lexicon_score(self, lexicon):
        score = 0
        for token in self.bag_ngrams[1]:
            if token in lexicon:
                score += self.bag_ngrams[1][token] * lexicon[token]
        if (self.rating * score > 0):
            return 1
        else:
            return 0

    def train_ngrams(self, freqs, ngram=1):
        for tok, freq in self.bag_ngrams[ngram].items():
            freqs[tok] += freq
    
    def tokenize(self):
        with open(self.path, 'r') as f:
            for line in f:
                for index, word in enumerate(line.split()):
                    split_word = space_punctuation(word)
                    if (index == 0):
                        self.first_in_sentence[split_word[0]] += 1
                    for seg in split_word:
                        if seg.upper() != seg:
                            seg = seg.lower()
                        self.bag_ngrams[1][seg] += 1
                        self.text.append(seg)
                        if seg in STOPWORDS:
                            self.stopwords += 1
        self.get_ngrams()
        
    def get_ngrams(self):
        bigrams = zip(self.text, self.text[1:])
        for token in bigrams:
            self.bag_ngrams[2][token] += 1

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

def get_stopwords(docs):
    stopwords = []
    pos_toks = collections.defaultdict(int)
    neg_toks = collections.defaultdict(int)
    info = collections.defaultdict(float)
    for doc in docs:
        num = int(os.path.basename(doc.path)[2:5])
        if num < 100:
            for tok in doc.bag_ngrams[1]:
                if POS in doc.path:
                    pos_toks[tok] += doc.bag_ngrams[1][tok]
                else:
                    neg_toks[tok] += doc.bag_ngrams[1][tok]
    pos_total = sum(pos_toks.values())
    neg_total = sum(pos_toks.values())
    vocab = set(pos_toks.keys()).union(set(neg_toks.keys()))
    for tok in vocab:
        info[tok] = (log(1 + pos_toks[tok] / pos_total)
                     + log(1 + neg_toks[tok] / neg_total)
                     - 2 * log((pos_toks[tok] + neg_toks[tok])
                               / (pos_total + neg_total)))
    sorted_info = sorted(info, key=lambda x: info[x])
    for tok in sorted_info[:100]:
        logging.info('{} {}'.format(tok, info[tok]))
    for tok in sorted_info[-100:]:
        logging.info('{} {}'.format(tok, info[tok]))
    logging.info('[{}]'.format("', '".join(sorted_info[:80])))
        
def tokenize_files(doc_dir, docs):
    doc_paths = walk_dir(doc_dir)
    for path in doc_paths:
        new_doc = Review(path)
        new_doc.tokenize()
        docs.append(new_doc)
    #get_stopwords(docs)
