#!/usr/bin/env python
from __future__ import division
import argparse
import collections
import os
import re
import string

STOPWORDS = set([',', '.', 'the', 'a', 'of', 'to', 'and', 'is', '"', 'in', "'s", 'that', 'it', ')', '(', 'with', 'I', 'as', 'for', 'film' 'this', 'his', 'her', 'their', 'they', 'film'])

GENERIC_PUNC = re.compile(r"(\w*)([{}])(\w*)".format(string.punctuation.replace('-','')))
POS = 'POS'
NEG = 'NEG'

class Review(object):
    def __init__(self, rating):
        self.rating = rating
        self.text = []
        self.score = 0

    def lexicon_score(self, lexicon):
        score = 0
        for token in self.text:
            if token in lexicon:
                score += lexicon[token]
        self.score = score
        if (self.rating * self.score > 0):
            return 1
        else:
            return 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    parser.add_argument('-l', '--lexicon', help='path to sentiment lexicon', default='resources/sent_lexicon')
    parser.add_argument('-w', '--weighted', action='store_true', help='flag to use magnitude weights from sentiment lexicon')
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

def get_review_files(review_dir, review_type, path_review_dict):
    search_dir = os.path.join(review_dir, review_type)
    review_paths = walk_dir(search_dir)
    rating = 1 if review_type == POS else -1
    for path in review_paths:
        path_review_dict[path] = Review(rating)

def tokenize(review_path, path_review_dict, sent_freqs):
    split_review = []
    with open(review_path, 'r') as f:
        for line in f:
            for word in line.split():
                split_word = space_punctuation(word)
                split_review.extend(split_word)
                for seg in split_word:
                    if seg not in STOPWORDS:
                        sent_freqs[seg] += 1
    path_review_dict[review_path].text = split_review

def get_sentiments(lex_path, weighted):
    '''
    Args:
    lex_path: string path to sentiment lexicon
    weighted: boolean, true if extracting sentiment magnitude as well as sign
    Returns:
    dictionary mapping word to sign, optionally weighted by estimated magnitude
    '''
    sent_lexicon = {}
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
            score = sign * weight if weighted else sign
            sent_lexicon[entry['word1']] = score
    return sent_lexicon

def sign_test(pos_count, neg_count):
    

if __name__ == '__main__':
    args = get_args()
    sent_lexicon = get_sentiments(args.lexicon, args.weighted)
    path_review_dict = dict()
    get_review_files(args.path, POS, path_review_dict)
    get_review_files(args.path, NEG, path_review_dict)
    pos_freqs = collections.defaultdict(int)
    neg_freqs = collections.defaultdict(int)
    correct_pos = correct_neg = 0
    for review_path, review in path_review_dict.items():
        if review.rating == 1:
            tokenize(review_path, path_review_dict, pos_freqs)
            correct_pos += review.lexicon_score(sent_lexicon)
        else:
            tokenize(review_path, path_review_dict, neg_freqs)
            correct_neg += review.lexicon_score(sent_lexicon)
    print correct_count, correct_count / len(path_review_dict)
    top = 100
    #print "Pos words: {}".format(sorted(pos_freqs, reverse=True, key=lambda x: pos_freqs[x])[:top])
    #print "Neg words: {}".format(sorted(neg_freqs, reverse=True, key=lambda x: neg_freqs[x])[:top])
