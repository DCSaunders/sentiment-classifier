#!/usr/bin/env python
from __future__ import division
import argparse
import codecs
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
import tokenizer
import lda
reload(sys)
sys.setdefaultencoding('utf-8')

POS = 'POS'
NEG = 'NEG'

class Freqs(object):
    def __init__(self):
        self.pos = collections.defaultdict(int)
        self.pos_stopwords = 0
        self.neg = collections.defaultdict(int)
        self.neg_stopwords = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    parser.add_argument('-l', '--lexicon', help='path to sentiment lexicon', default='resources/sent_lexicon')
    parser.add_argument('-N', '--cv_folds', help='number of folds for N-fold cross-validation', default=10)
    parser.add_argument('-K', '--topic_count', help='number of topics for LDA', default=10)
    args = parser.parse_args()
    return args

def get_review_files(review_dir, review_type, reviews, vocab):
    search_dir = os.path.join(review_dir, review_type)
    tokenizer.tokenize_files(search_dir, reviews, vocab)

def get_sentiments(lex_path):
    '''
    Args:
    lex_path: string path to sentiment lexicon
    Returns:
    dictionaries mapping word to sign, one weighted, one unweighted
    '''
    unweight_lex, weight_lex = {}, {}
    with open(lex_path, 'r') as f:
        for line in f:
            entries = [value.split('=') for value in line.split()]
            entry = {value[0]: value[1] for value in entries}
            sign_text = entry['priorpolarity']
            if sign_text == 'positive':
                sign = 1
            elif sign_text == 'neutral':
                sign = 0
            else:
                sign = -1
            weight = 1 if entry['type'] == 'strongsubj' else 0.5
            weight_lex[entry['word1']] = sign * weight
            unweight_lex[entry['word1']] = sign
    return unweight_lex, weight_lex

def sign_test(results, label_1, label_2):
    test1_count = test2_count = 0
    shared_obs = set(results[label_1]).intersection(results[label_2])
    for obs in shared_obs:
        if results[label_1][obs] == results[label_2][obs]:
            # Add half to each for ties
            test1_count += 0.5
            test2_count += 0.5
        elif results[label_1][obs] > results[label_2][obs]:
            test1_count += 1
        elif results[label_1][obs] < results[label_2][obs]:
            test2_count += 1
    significance = two_sided_binomial(round(test1_count), round(test2_count))
    print "Comparing {} and {}: equal with probability {}".format(
        label_1, label_2, significance)

def two_sided_binomial(test1, test2):
    # normal approximation: binom(p) -> N(np, np^2)
    mean = (test1 + test2) * 0.5
    std = np.sqrt(mean * 0.5)
    corrected_point = min(test1, test2) + 0.5 # Continuity correction
    approx = 2 * norm.cdf(corrected_point, loc=mean, scale=std)
    return approx
       
def naive_bayes(review, freqs, results, inc_stopwords=True, smooth=1.0,
                ngram=1):
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values())
    total_neg = sum(freqs.neg.values())
    if not inc_stopwords:
        total_pos -= freqs.pos_stopwords
        total_neg -= freqs.neg_stopwords
    for tok, freq in review.bag_ngrams[ngram].items():
        if inc_stopwords or not tok in tokenizer.STOPWORDS:
            pos_prob += freq * (log(freqs.pos[tok] + smooth)
                                - log((1 + smooth) * total_pos))
            neg_prob += freq * (log(freqs.neg[tok] + smooth)
                                - log((1 + smooth) * total_neg))
    if pos_prob == neg_prob:
            pos_prob, neg_prob = np.random.rand(2)
    if (pos_prob - neg_prob) * review.rating > 0.0:
        results[review] = 1
    else:
        results[review] = 0
                
def train(train_reviews, results):
    unigrams = Freqs()
    bigrams = Freqs()
    trigrams = Freqs()
    for review in train_reviews:
        if review.rating == 1:
            review.train_ngrams(unigrams.pos, ngram=1)
            review.train_ngrams(bigrams.pos, ngram=2)
            review.train_ngrams(trigrams.pos, ngram=3)
            unigrams.pos_stopwords += review.stopwords
        else:
            review.train_ngrams(unigrams.neg, ngram=1)
            review.train_ngrams(bigrams.neg, ngram=2)
            review.train_ngrams(trigrams.neg, ngram=3)
            unigrams.neg_stopwords += review.stopwords
    return unigrams, bigrams, trigrams

def split_train_test(reviews, low, high):
    train, test = [], []
    for review in reviews:
        num = int(os.path.basename(review.path)[2:5])
        if (num >= low) and (num < high):
            test.append(review)
        else:
            train.append(review)
    return train, test

def test(tests, unigrams, bigrams, trigrams, results):
    for review in tests:
        naive_bayes(review, unigrams, results['n_bayes'], smooth=0.0)
        naive_bayes(review, unigrams, results['bayes_smooth'])
        naive_bayes(review, unigrams, results['bayes_smooth_stopwords'],
                    inc_stopwords=False)
        naive_bayes(review, bigrams, results['bayes_bg'], ngram=2)
        naive_bayes(review, trigrams, results['bayes_tg'], ngram=3)

def cross_validate(reviews, results, cv_folds):
    count = len(reviews) / 2 # assume equal number pos/neg reviews
    labels = ['n_bayes', 'bayes_smooth', 'bayes_smooth_stopwords', 'bayes_bg', 'bayes_tg']
    fold_size = count / cv_folds
    accuracies = collections.defaultdict(list)
    for fold in range(cv_folds):
        low = fold * fold_size
        high = (fold + 1) * fold_size
        train_reviews, test_reviews = split_train_test(
            reviews, low, high)
        unigrams, bigrams, trigrams = train(train_reviews, results)
        test(test_reviews, unigrams, bigrams, trigrams, results)
        #sign_test(results, 'n_bayes', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'uw_lex')
        #sign_test(results, 'bayes_smooth', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'n_bayes')
        #sign_test(results, 'bayes_smooth', 'bayes_smooth_stopwords')
        for label in labels:
            accuracy = sum(results[label].values()) / len(results[label])
            print label, accuracy
            accuracies[label].append(accuracy)
            results[label] = {}
    print accuracies

def lexicon_test(reviews, unweight_lex, weight_lex, results):
    for review in reviews:
        results['uw_lex'][review] = review.lexicon_score(unweight_lex)
        results['w_lex'][review] = review.lexicon_score(weight_lex)
    print 'uw_lex', (sum(results['uw_lex'].values())
                     / len(results['uw_lex']))
    print 'w_lex', (sum(results['w_lex'].values())
                    / len(results['w_lex']))
    sign_test(results, 'w_lex', 'uw_lex')
        
def output_topics(topics, top_words):
    for topic in topics:
        words = topic.word_counts.keys()
        counts = np.array([topic.word_counts[w] for w in words])
        top = np.argpartition(counts, -top_words)[-top_words:]
        top = top[np.argsort(counts[top])]
        print [words[ind] for ind in top]
                
if __name__ == '__main__':
    np.random.seed(1234)
    args = get_args()
    unweight_lex, weight_lex = get_sentiments(args.lexicon)
    reviews = []
    results = {'w_lex': {}, 'uw_lex': {}, 'n_bayes': {},
               'bayes_smooth': {}, 'bayes_smooth_stopwords': {},
               'bayes_bg': {}, 'bayes_tg': {}}
    vocab = set()
    get_review_files(args.path, POS, reviews, vocab)
    get_review_files(args.path, NEG, reviews, vocab)
    lexicon_test(reviews, unweight_lex, weight_lex, results)
    cross_validate(reviews, results, args.cv_folds)
