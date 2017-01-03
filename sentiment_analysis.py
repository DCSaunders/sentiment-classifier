#!/usr/bin/env python
from __future__ import division
import argparse
import codecs
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
import tokenizer
import lda
import slda
reload(sys)
sys.setdefaultencoding('utf-8')

POS = 'POS'
NEG = 'NEG'

class Freqs(object):
    def __init__(self):
        self.pos = collections.defaultdict(int)
        self.neg = collections.defaultdict(int)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='data',
        help='path to dir with POS and NEG review subdirs')
    parser.add_argument('-l', '--lexicon', help='path to sentiment lex',
        default='resources/sent_lexicon')
    parser.add_argument('-N', '--cv_folds', type=int, default=10,
        help='number of folds for N-fold cross-validation')
    parser.add_argument('-K', '--topic_count', type=int,
        help='number of topics for LDA', default=10)
    parser.add_argument('--no_single_doc_toks', default=False,
        action='store_true',
        help='Set if removing tokens only in one training doc')
    parser.add_argument('--get_mi_stopwords', default=False,
        action='store_true',
        help='Set if getting stopwords via MI from training reviews')
    parser.add_argument('--run_slda', default=False,
        action='store_true', help='Set if running supervised LDA')
    parser.add_argument('-t', '--train_iters', type=int,
        help='training iterations for (s)LDA', default=20)
    parser.add_argument('-a', '--alpha', type=float,
        help='alpha for (s)LDA', default=0.1)
    parser.add_argument('-g', '--gamma', type=float,
        help='gamma for (s)LDA', default=0.1)
    args = parser.parse_args()
    return args

def get_review_files(review_dir, reviews):
    # Get all review files and complete vocab counts
    for review_type in (POS, NEG):
        search_dir = os.path.join(review_dir, review_type)
        tokenizer.tokenize_files(search_dir, reviews)
        
def preprocess_reviews(train_reviews, test_reviews, no_single_doc=False,
                       get_mi_stopwords=False):
    vocab = collections.Counter()
    doc_occurrences = collections.defaultdict(int)
    if get_mi_stopwords:
        stopwords = tokenizer.get_stopwords(train_reviews)
        logging.info('Stopwords: {}'.format(stopwords))
    else:
        stopwords = tokenizer.STOPWORDS
    for review in train_reviews:
        for w, count in review.bag_ngrams[1].items():
            vocab[w] += count
            doc_occurrences[w] += 1
    common_vocab = set([w for w, count in vocab.items()
                    if count > 2 # no rare words
                        and w not in stopwords]) # no stopwords
    if no_single_doc:
        gt_one_doc = set([w for w, count in doc_occurrences.items()
                          if count > 1])
        common_vocab = common_vocab.intersection(gt_one_doc)

    # Apply trained vocab to all reviews, train and test
    for review in train_reviews + test_reviews:
        review.text_no_stopwords = []
        for seg in review.text:
            if seg in common_vocab:
                review.text_no_stopwords.append(seg)
    
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
    logging.info("Comparing {} and {}: equal with probability {}".format(
        label_1, label_2, significance))

def two_sided_binomial(test1, test2):
    # normal approximation: binom(p) -> N(np, np^2)
    mean = (test1 + test2) * 0.5
    std = np.sqrt(mean * 0.5)
    corrected_point = min(test1, test2) + 0.5 # Continuity correction
    approx = min(2 * norm.cdf(corrected_point, loc=mean, scale=std), 1.0)
    return approx
       
def naive_bayes(review, freqs, results, smooth=1.0, ngram=1):
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values())
    total_neg = sum(freqs.neg.values())
    for tok, freq in review.bag_ngrams[ngram].items():
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
    for review in train_reviews:
        if review.rating == 1:
            review.train_ngrams(unigrams.pos, ngram=1)
            review.train_ngrams(bigrams.pos, ngram=2)
        else:
            review.train_ngrams(unigrams.neg, ngram=1)
            review.train_ngrams(bigrams.neg, ngram=2)
    return unigrams, bigrams

def split_train_test(reviews, low, high):
    train, test = [], []
    for review in reviews:
        num = int(os.path.basename(review.path)[2:5])
        if (num >= low) and (num < high):
            test.append(review)
        else:
            train.append(review)
    return train, test

def test(tests, unigrams, bigrams, results):
    for review in tests:
        naive_bayes(review, unigrams, results['n_bayes'], smooth=0.0)
        naive_bayes(review, unigrams, results['bayes_smooth'])
        naive_bayes(review, bigrams, results['bayes_bg'], ngram=2)

def cross_validate(reviews, results, args):
    count = len(reviews) / 2 # assume equal number pos/neg reviews
    labels = ['n_bayes', 'bayes_smooth', 'bayes_bg', 'lda', 'slda']
    fold_size = count / args.cv_folds
    accuracies = collections.defaultdict(list)
    for fold in range(args.cv_folds):
        low = fold * fold_size
        high = (fold + 1) * fold_size
        train_reviews, test_reviews = split_train_test(reviews, low, high)
        preprocess_reviews(train_reviews, test_reviews, args.no_single_doc_toks,
                           args.get_mi_stopwords)
        run_lda(train_reviews, test_reviews, results['lda'],
                args.topic_count, args.train_iters, args.alpha,
                args.gamma)
        run_slda(train_reviews, test_reviews,
            results['slda'],  args.topic_count, args.train_iters,
            args.alpha, args.gamma)
        unigrams, bigrams = train(train_reviews, results)
        test(test_reviews, unigrams, bigrams, results)
        #sign_test(results, 'n_bayes', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'uw_lex')
        #sign_test(results, 'bayes_smooth', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'n_bayes')
        sign_test(results, 'bayes_smooth', 'bayes_bg')
        sign_test(results, 'bayes_smooth', 'lda')
        sign_test(results, 'bayes_smooth', 'slda')
        sign_test(results, 'slda', 'lda')
        for label in labels:
            if results[label]:
                accuracy = sum(results[label].values()) / len(results[label])
                logging.info('{}: {}'.format(label, accuracy))
                accuracies[label].append(accuracy)
                results[label] = {}
    logging.info(accuracies)

def run_slda(train_reviews, test_reviews, results, topic_count, train_iters, alpha, gamma):
    slda_results = slda.run_slda(train_reviews,
        test_reviews, topic_count, train_iters, alpha, gamma)
    pos_topics = np.zeros(topic_count)
    neg_topics = np.zeros(topic_count)
    for r in train_reviews:
        if r.rating == 1:
            pos_topics += r.topic_counts
        else:
            neg_topics += r.topic_counts
    for r, val in slda_results.items():
        results[r] = val
    logging.info('sLDA pos topics: {}\n sLDA neg topics: {}'.format(
        pos_topics / np.sum(pos_topics),
        neg_topics / np.sum(neg_topics)))
    
def run_lda(train_reviews, test_reviews, results, topic_count, train_iters, alpha, gamma):
    lda.run_lda(train_reviews, test_reviews, topic_count,
                train_iters, alpha, gamma)
    pos_topics = np.zeros(topic_count)
    neg_topics = np.zeros(topic_count)
    for r in train_reviews:
        if r.rating == 1:
            pos_topics += r.topic_counts
        else:
            neg_topics += r.topic_counts
    total_pos = np.sum(pos_topics)
    total_neg = np.sum(neg_topics)
    smooth = 1
    logging.info('LDA pos topics: {} \n LDA neg topics: {}'.format(
        pos_topics / np.sum(pos_topics),
        neg_topics / np.sum(neg_topics)))
    for r in test_reviews:
        neg_prob = pos_prob = 0.0
        for index, count in enumerate(r.topic_counts):
            pos_prob += count * (log(pos_topics[index] + smooth)
                                - log((1 + smooth) * total_pos))
            neg_prob += count * (log(neg_topics[index] + smooth)
                                - log((1 + smooth) * total_neg))
        if pos_prob == neg_prob:
            logging.info(
                'Equal probabilities - choose class at random')
            pos_prob, neg_prob = np.random.rand(2)
        if (pos_prob - neg_prob) * r.rating > 0.0:
            results[r] = 1
        else:
            results[r] = 0

def lexicon_test(reviews, unweight_lex, weight_lex, results):
    for review in reviews:
        results['uw_lex'][review] = review.lexicon_score(unweight_lex)
        results['w_lex'][review] = review.lexicon_score(weight_lex)
    logging.info('uw_lex {}'.format(sum(results['uw_lex'].values())
                     / len(results['uw_lex'])))
    logging.info('w_lex {}'.format(sum(results['w_lex'].values())
                    / len(results['w_lex'])))
    sign_test(results, 'w_lex', 'uw_lex')
        
def output_topics(topics, top_words):
    for topic in topics:
        words = topic.word_counts.keys()
        counts = np.array([topic.word_counts[w] for w in words])
        top = np.argpartition(counts, -top_words)[-top_words:]
        top = top[np.argsort(counts[top])]
        logging.info([words[ind] for ind in top])
                
if __name__ == '__main__':
    np.random.seed(1234)
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    #unweight_lex, weight_lex = get_sentiments(args.lexicon)
    reviews = []
    results = {'w_lex': {}, 'uw_lex': {}, 'n_bayes': {},
               'bayes_smooth': {}, 'bayes_bg': {}, 'lda': {}, 'slda': {}}
    get_review_files(args.path, reviews)
    #lexicon_test(reviews, unweight_lex, weight_lex, results)
    cross_validate(reviews, results, args)
