#!/usr/bin/env python
from __future__ import division
import argparse
import codecs
import collections
import copy
import os
import re
import string
import sys
from numpy import log
from scipy.stats import binom_test
reload(sys)
sys.setdefaultencoding('utf-8')

STOPWORDS = set([',', '.', 'the', 'a', 'of', 'to', 'and', 'is', '"', 'in', "'s", 'that', 'it', ')', '(', 'with', 'I', 'as', 'for', 'film' 'this', 'his', 'her', 'their', 'they', 'film'])

GENERIC_PUNC = re.compile(r"(\w*-?\w*)(--|\.\.\.|[,!?%`./();$&@#:\"'])(\w*-?\w*)") 

POS = 'POS'
NEG = 'NEG'

class Freqs(object):
    def __init__(self):
        self.pos = collections.defaultdict(int)
        self.neg = collections.defaultdict(int)
        self.pos_stopwords = 0
        self.neg_stopwords = 0
        
class Review(object):
    def __init__(self, rating, path):
        self.rating = rating
        self.path = path
        self.text = []
        self.bag_ngrams = {1: collections.defaultdict(int),
                           2: collections.defaultdict(int),
                           3: collections.defaultdict(int)}
        self.first_in_sentence = collections.defaultdict(int)
        self.stopwords = 0
        
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
    
    def tokenize(self):
        with codecs.open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                for index, word in enumerate(line.split()):
                    split_word = space_punctuation(word)
                    self.text.extend(split_word)
                    if (index == 0):
                        self.first_in_sentence[split_word[0]] += 1
                    for seg in split_word:
                        self.bag_ngrams[1][seg] += 1
                        if seg in STOPWORDS:
                            self.stopwords += 1
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
    parser.add_argument('-l', '--lexicon', help='path to sentiment lexicon', default='resources/sent_lexicon')
    parser.add_argument('-N', '--cv_folds', help='number of folds for N-fold cross-validation', default=10)
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

def walk_dir(review_dir):
    review_list = []
    for dirpath, dirnames, filenames in os.walk(review_dir):
        for filename in filenames:
            review_list.append(os.path.join(dirpath, filename))
    return review_list

def get_review_files(review_dir, review_type, reviews):
    search_dir = os.path.join(review_dir, review_type)
    review_paths = walk_dir(search_dir)
    rating = 1 if review_type == POS else -1
    for path in review_paths:
        new_review = Review(rating, path)
        new_review.tokenize()
        reviews.append(new_review)

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
    return binom_test((test1, test2), p=0.5, alternative='two-sided')
    
def naive_bayes_recased(review, freqs, results, smooth=1.0):
    # Naive Bayes with optional smoothing and recasing.
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values())
    total_neg = sum(freqs.neg.values())
    recase = collections.defaultdict(int)
    for tok, freq in review.first_in_sentence.items():
        recase[tok] = freq
    for word, freq in review.bag_ngrams[1].items():
        if recase[word] > 0: # lowercase instances accounted for separately
            word = word.lower() 
            recase[word] -= freq
        pos_prob += freq * (log(freqs.pos[word] + smooth)
                     - log((1 + smooth) * total_pos))
        neg_prob += freq * (log(freqs.neg[word] + smooth)
                     - log((1 + smooth) * total_neg))
    if (pos_prob - neg_prob) * review.rating > 0.0:
        results[review] = 1
    else:
        results[review] = 0

def naive_bayes(review, freqs, results, smooth=1.0, ngram=1):
    # Naive Bayes with optional smoothing.
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values())
    total_neg = sum(freqs.neg.values())
    for tok, freq in review.bag_ngrams[ngram].items():
        pos_prob += freq * (log(freqs.pos[tok] + smooth)
                     - log((1 + smooth) * total_pos))
        neg_prob += freq * (log(freqs.neg[tok] + smooth)
                     - log((1 + smooth) * total_neg))
    if (pos_prob - neg_prob) * review.rating > 0.0:
        results[review] = 1
    else:
        results[review] = 0
                

def naive_bayes_stopwords(review, freqs, results, smooth=1.0):
    # Naive Bayes with optional smoothing and stopwords.
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values()) - freqs.pos_stopwords
    total_neg = sum(freqs.neg.values()) - freqs.neg_stopwords
    for word, freq in review.bag_ngrams[1].items():   
        if not word in STOPWORDS:
           pos_prob += freq * (log(freqs.pos[word] + smooth)
                        - log((1 + smooth) * total_pos))
           neg_prob += freq * (log(freqs.neg[word] + smooth)
                        - log((1 + smooth) * total_neg))
    if (pos_prob - neg_prob) * review.rating > 0.0:
        results[review] = 1
    else:
        results[review] = 0


def train(train_reviews, results):
    unigrams = Freqs()
    bigrams = Freqs()
    trigrams = Freqs()
    to_recase = Freqs()
    recased_freqs = Freqs()
    for review in train_reviews:
        if review.rating == 1:
            review.train_ngrams(unigrams.pos, to_recase.pos,
                                recase=True, ngram=1)
            review.train_ngrams(bigrams.pos, None, ngram=2)
            review.train_ngrams(trigrams.pos, None, ngram=3)
            unigrams.pos_stopwords += review.stopwords
        else:
            review.train_ngrams(unigrams.neg, to_recase.neg,
                                recase=True, ngram=1)
            review.train_ngrams(bigrams.neg, None, ngram=2)
            review.train_ngrams(trigrams.neg, None, ngram=3)
            unigrams.neg_stopwords += review.stopwords
    recased_freqs.pos = recase(to_recase.pos, unigrams.pos)
    recased_freqs.neg = recase(to_recase.neg, unigrams.neg)
    return unigrams, bigrams, trigrams, recased_freqs


def recase(to_recase, freqs):
    recased_freqs = collections.defaultdict(int)
    for tok, freq in freqs.items():
        if to_recase[tok] == freq:
            recased_freqs[tok.lower] = freq + freqs[tok.lower()]
        else:
            to_recase.pop(tok)
            recased_freqs[tok] = freq
    return recased_freqs

def split_train_test(reviews, low, high):
    train, test = [], []
    for review in reviews:
        num = int(os.path.basename(review.path)[2:5])
        if (num >= low) and (num < high):
            test.append(review)
        else:
            train.append(review)
    return train, test

def test(tests, unigrams, bigrams, trigrams, recased_freqs, results):
    for review in tests:
        naive_bayes(review, unigrams, results['n_bayes'], smooth=0.0)
        naive_bayes(review, unigrams, results['bayes_smooth'], smooth=1.0)
        naive_bayes_recased(
            review, recased_freqs, results['bayes_recased'], smooth=1.0)
        naive_bayes_stopwords(review, unigrams,
                              results['bayes_smooth_stopwords'], smooth=1.0)
        naive_bayes(review, bigrams, results['bayes_bg'], smooth=1.0, ngram=2)
        naive_bayes(review, trigrams, results['bayes_tg'], smooth=1.0, ngram=3)


def cross_validate(reviews, results, cv_folds):
    count = len(reviews) / 2 # assume equal number pos/neg reviews
    labels = ['n_bayes', 'bayes_smooth', 'bayes_recased', 'bayes_smooth_stopwords', 'bayes_bg', 'bayes_tg']
    fold_size = count / cv_folds
    accuracies = collections.defaultdict(list)
    for fold in range(cv_folds):
        low = fold * fold_size
        high = (fold + 1) * fold_size
        train_reviews, test_reviews = split_train_test(reviews, low, high)
        unigrams, bigrams, trigrams, recased_freqs = train(
            train_reviews, results)
        test(test_reviews, unigrams, bigrams, trigrams, recased_freqs, results)
        #sign_test(results, 'n_bayes', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'uw_lex')
        #sign_test(results, 'bayes_smooth', 'w_lex')
        #sign_test(results, 'bayes_smooth', 'n_bayes')
        #sign_test(results, 'bayes_smooth', 'bayes_smooth_stopwords')
        #sign_test(results, 'bayes_smooth', 'bayes_recased')
        #sign_test(results, 'bayes_recased', 'bayes_smooth_stopwords')
        #sign_test(results, 'bayes_recased', 'w_lex')
        #sign_test(results, 'bayes_recased', 'uw_lex')
        #sign_test(results, 'bayes_recased', 'uw_lex')
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
    print 'uw_lex', sum(results['uw_lex'].values()) / len(results['uw_lex'])
    print 'w_lex', sum(results['w_lex'].values()) / len(results['w_lex'])
    sign_test(results, 'w_lex', 'uw_lex')
        
        
if __name__ == '__main__':
    args = get_args()
    unweight_lex, weight_lex = get_sentiments(args.lexicon)
    reviews = []
    results = {'w_lex': {}, 'uw_lex': {}, 'n_bayes': {},
               'bayes_smooth': {}, 'bayes_smooth_stopwords': {},
               'bayes_recased': {}, 'bayes_bg': {}, 'bayes_tg': {}}
    get_review_files(args.path, POS, reviews)
    get_review_files(args.path, NEG, reviews)
    lexicon_test(reviews, unweight_lex, weight_lex, results)
    cross_validate(reviews, results, args.cv_folds)
