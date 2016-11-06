#!/usr/bin/env python
import argparse
import collections
import os

def get_review_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='path to dir with POS and NEG review subdirs', default='data')
    args = parser.parse_args()
    return args.path

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
                # TODO: want to maintain apostrophes, not get rid of them
                split_word = word.split("'")
                if (len(split_word) > 1):
                    if word[-3:] == "n't":
                        split_word = [word[:-3], word[-3:]]
                for seg in split_word:
                    split_review.append(seg)
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

    print "Pos words: {}".format(pos_freqs)
    print "Neg words: {}".format(neg_freqs)
    
