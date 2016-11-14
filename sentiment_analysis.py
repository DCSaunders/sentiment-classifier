#!/usr/bin/env python
from __future__ import division
import argparse
import collections
import os
import re
import string
from numpy import log
from scipy.stats import binom_test

STOPWORDS = set([',', '.', 'the', 'a', 'of', 'to', 'and', 'is', '"', 'in', "'s", 'that', 'it', ')', '(', 'with', 'I', 'as', 'for', 'film' 'this', 'his', 'her', 'their', 'they', 'film'])

GENERIC_PUNC = re.compile(r"(\w*)([{}])(\w*)".format(string.punctuation.replace('-','')))
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
        self.bag_words = {}

    def lexicon_score(self, lexicon):
        score = 0
        for token in self.text:
            if token in lexicon:
                score += lexicon[token]
        if (self.rating * score > 0):
            return 1
        else:
            return 0
        
    def test_tokenize(self):
        split_review = []
        with open(self.path, 'r') as f:
            for line in f:
                for word in line.split():
                    split_word = space_punctuation(word)
                    split_review.extend(split_word)
        self.text = split_review

    def train_tokenize(self, sent_freqs):
        split_review = []
        stopword_count = 0
        with open(self.path, 'r') as f:
            for line in f:
                for index, word in enumerate(line.split()):
                    split_word = space_punctuation(word)
                    split_review.extend(split_word)
                    for seg in split_word:
                        sent_freqs[seg] += 1
                        if seg in STOPWORDS:
                            stopword_count += 1
        self.text = split_review
        return stopword_count

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    parser.add_argument('-l', '--lexicon', help='path to sentiment lexicon', default='resources/sent_lexicon')
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
        reviews.append(Review(rating, path))

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
    for obs in results[label_1]:
        if obs in results[label_2]:
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
    
def naive_bayes(review, freqs, results, smooth=1.0):
    # Naive Bayes with optional smoothing.
    # Assume equal class priors: P(neg) = P(pos) = 0.5
    neg_prob = pos_prob = 0.0
    total_pos = sum(freqs.pos.values())
    total_neg = sum(freqs.neg.values())
    for word in review.text:
        if (smooth > 0.0 or (freqs.pos[word] and freqs.neg[word])):
            pos_prob += (log(freqs.pos[word] + smooth)
                         - log((1 + smooth) * total_pos))
            neg_prob += (log(freqs.neg[word] + smooth)
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
    for word in review.text:
        if (smooth > 0.0 or (pos_freqs[word] and neg_freqs[word])):
            if not word in STOPWORDS:
                pos_prob += (log(freqs.pos[word] + smooth)
                             - log((1 + smooth) * total_pos))
                neg_prob += (log(freqs.neg[word] + smooth)
                             - log((1 + smooth) * total_neg))
    if (pos_prob - neg_prob) * review.rating > 0.0:
        results[review] = 1
    else:
        results[review] = 0


def train(train_reviews, results):
    freqs = Freqs()
    for review in train_reviews:
        if review.rating == 1:
            freqs.pos_stopwords += review.train_tokenize(freqs.pos)
        else:
            freqs.neg_stopwords += review.train_tokenize(freqs.neg)
        results['uw_lex'][review] = review.lexicon_score(unweight_lex)
        results['w_lex'][review] = review.lexicon_score(weight_lex)
    return freqs

def split_train_test(reviews, low, high):
    train, test = [], []
    for review in reviews:
        num = int(os.path.basename(review.path)[2:5])
        if (num >= low) and (num < high):
            train.append(review)
        else:
            test.append(review)
    return train, test

def test(tests, freqs, unweight_lex, weight_lex, results):
    for review in tests:
        review.test_tokenize()
        naive_bayes(review, freqs, results['n_bayes'], smooth=0.0)
        naive_bayes(review, freqs, results['bayes_smooth'], smooth=1.0)
        naive_bayes_stopwords(review, freqs,
                              results['bayes_smooth_stopwords'], smooth=1.0)
        results['uw_lex'][review] = review.lexicon_score(unweight_lex)
        results['w_lex'][review] = review.lexicon_score(weight_lex)
    
if __name__ == '__main__':
    args = get_args()
    unweight_lex, weight_lex = get_sentiments(args.lexicon)
    reviews = []
    get_review_files(args.path, POS, reviews)
    get_review_files(args.path, NEG, reviews)
    results = {'w_lex': {}, 'uw_lex': {}, 'n_bayes': {},
               'bayes_smooth': {}, 'bayes_smooth_stopwords': {}}
    train_reviews, test_reviews = split_train_test(reviews, low=0, high=900)
    freqs = train(train_reviews, results)
    test(test_reviews, freqs, unweight_lex, weight_lex, results)
    for result in results:
        print result, sum(results[result].values()) / len(results[result])
    sign_test(results, 'w_lex', 'uw_lex')
    sign_test(results, 'n_bayes', 'w_lex')
    sign_test(results, 'bayes_smooth', 'n_bayes')
    sign_test(results, 'bayes_smooth', 'bayes_smooth_stopwords')
    
