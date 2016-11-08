#!/usr/bin/env python
import argparse
import collections
import os
import re
import string

GENERIC_PUNC = re.compile(r"(\w*)([{}])(\w*)".format(string.punctuation.replace('-','')))

def get_review_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    args = parser.parse_args()
    return args.path

def space_punctuation(word):
    #line = re.sub(GENERIC_PUNC, " ", line)
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
            
def get_review_files(review_dir):
    pos_dir = os.path.join(review_dir, 'POS')
    neg_dir = os.path.join(review_dir, 'NEG')
    pos_reviews = walk_dir(pos_dir)
    neg_reviews = walk_dir(neg_dir)
    return pos_reviews, neg_reviews

def tokenize(review, sentiment_freqs):
    split_review = []
    with open(review, 'r') as f:
        for line in f:
            for word in line.split():
                split_word = space_punctuation(word)
                split_review.extend(split_word)
                for seg in split_word:
                    sentiment_freqs[seg] += 1
    return split_review
    
if __name__ == '__main__':
    review_dir = get_review_dir()
    pos_reviews, neg_reviews = get_review_files(review_dir)
    pos_freqs = collections.defaultdict(int)
    neg_freqs = collections.defaultdict(int)
    split_reviews = []
    for review in pos_reviews:
        split_reviews.append(tokenize(review, pos_freqs))
    for review in neg_reviews:
        split_reviews.append(tokenize(review, neg_freqs))
    top = 100
    print "Pos words: {}".format(sorted(pos_freqs, reverse=True, key=lambda x: pos_freqs[x])[:top])
    print "Neg words: {}".format(sorted(neg_freqs, reverse=True, key=lambda x: neg_freqs[x])[:top])

