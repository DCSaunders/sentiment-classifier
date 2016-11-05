#!/usr/bin/env python
import argparse
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

if __name__ == '__main__':
    review_dir = get_review_dir()
    pos_reviews, neg_reviews = get_review_files(review_dir)
    for review in pos_reviews:
        with open(review, 'r') as f:
            for line in f:
                print line
