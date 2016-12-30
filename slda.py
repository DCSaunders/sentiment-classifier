from __future__ import division

from collections import defaultdict
import numpy as np
import tokenizer
import logging

class Topic(object):
    def __init__(self):
        # Store counts of words given topic, across all documents
        self.word_counts = defaultdict(int)
        
def initialise(docs, topics, vocab, topic_word_assign, K):
    for doc in docs:
        doc.topic_words = []
        doc.topic_counts = np.zeros(K)
        for word in doc.text_no_stopwords:
            t = np.random.randint(0, K)
            doc.topic_words.append(t)
            doc.topic_counts[t] += 1
            topics[t].word_counts[word] += 1
    for topic_idx, topic in enumerate(topics):
        topic_word_assign[topic_idx] = sum(topic.word_counts.values())
    words_given_topics = defaultdict(int)
    for w in vocab:
        words_given_topics[w] = np.array([t.word_counts[w] for t in topics])
    return words_given_topics

def rating_to_regr(rating, reverse=False):
    if not reverse:
        # regression targets are 0 and 1
        return (rating + 1) / 2
    else:
        return (rating - 0.5) * 2
           
def train(docs, topics, train_iters, vocab_size,
          topic_word_assign, words_given_topics, eta,
          alpha=0.1, gamma=0.1):
    K = len(topics)
    sample_reg_every = 100
    labels = np.array([rating_to_regr(doc.rating) for doc in docs])
    
    for i in range(0, train_iters):
        logging.info('Iteration {}'.format(i))
        for doc_index, doc in enumerate(docs):
            mu = sum(doc.topic_counts * eta)
            for index, word in enumerate(doc.text_no_stopwords):
                old_topic = doc.topic_words[index]    
                topics[old_topic].word_counts[word] -= 1
                topic_word_assign[old_topic] -= 1
                doc.topic_counts[old_topic] -= 1
                words_given_topics[word][old_topic] -= 1
                mu -= eta[old_topic]
                y = labels[doc_index]
                y_diff = y - (mu + eta) / len(docs)
                distrib = ((alpha + doc.topic_counts)
                           * (gamma + words_given_topics[word])
                           * np.exp(-0.5 * np.square(y_diff))
                           / (vocab_size * gamma + topic_word_assign))
                new_topic = sample_discrete(distrib)
                doc.topic_words[index] = new_topic
                doc.topic_counts[new_topic] += 1
                topics[new_topic].word_counts[word] += 1
                topic_word_assign[new_topic] += 1
                words_given_topics[word][new_topic] += 1
                mu += eta[new_topic]
            if doc_index % sample_reg_every == 0:
                # resample eta
                topic_probs = np.array(
                    [doc.topic_counts / sum(doc.topic_counts) for doc in docs])
                precision = topic_probs.T.dot(topic_probs) + np.eye(K)
                eta = np.linalg.lstsq(precision, topic_probs.T.dot(labels))
                
def sample_discrete(distribution):
    r = sum(distribution) * np.random.uniform()
    total = 0
    for choice, p in enumerate(distribution):
        total += p
        if total >= r:
            return choice

def run_slda(train_docs, test_docs, K, train_iters=100):
    top_words = 10
    topics = []
    test_topics = []
    alpha = 0.1 # dirichlet parameter over topics
    gamma = 0.1 # dirichlet parameter over words
    eta = np.random.randn(K) # regression coefficients for topics
    test_samples = 10
    vocab = set()
    for review in train_docs:
        vocab = vocab.union(review.text_no_stopwords)
    vocab_size = len(vocab)
    logging.info(
        'sLDA with vocab size {}, {} training iterations, {} topics'.format(
            vocab_size, train_iters, K))
    for t in range(0, K):
        topics.append(Topic())
        test_topics.append(Topic())
    topic_word_assign = np.zeros(K)
    words_given_topics = initialise(train_docs, topics, vocab,
                                    topic_word_assign, K)
    train(train_docs, topics, train_iters, vocab_size, topic_word_assign,
          words_given_topics, eta)
    test_topic_word_assign = np.zeros(K)
    initialise(test_docs, test_topics, vocab, test_topic_word_assign, K)
    
    for doc in test_docs:
        for i in range(0, train_iters):
            for index, word in enumerate(doc.text_no_stopwords):
                old_topic = doc.topic_words[index]
                doc.topic_counts[old_topic] -= 1
                distrib = ((alpha + doc.topic_counts)
                           * (gamma + words_given_topics[word])
                           / (vocab_size * gamma + topic_word_assign))
                new_topic = sample_discrete(distrib)
                doc.topic_words[index] = new_topic
                doc.topic_counts[new_topic] += 1
        topic_probs = doc.topic_counts / sum(doc.topic_counts)
        y_mu = sum(topic_probs * eta)
        samples = np.random.randn(test_samples) + y_mu
        logging.info('true rating {}, samples {}, y_mu {}'.format(
            rating_to_regr(doc.rating), samples), y_mu)
    for index, topic in enumerate(topics):
        top_topic_words = sorted(topic.word_counts,
                                 key=lambda x: topic.word_counts[x],
                                 reverse=True)[:top_words]
        logging.info('{}: {}'.format(index, ' '.join(top_topic_words)))
    
if __name__ == '__main__':
    np.random.seed(1234)
    # POS test dataset is sci.space
    train_reviews = []
    test_reviews = []
    test_count = 50
    tokenizer.tokenize_files('tmp/POS', train_reviews)
    test_reviews = train_reviews[-test_count:]
    train_reviews = train_reviews[:-test_count]
    # NEG test dataset is sci.med
    tokenizer.tokenize_files('tmp/NEG', train_reviews)
    test_reviews.extend(train_reviews[-test_count:])
    train_reviews = train_reviews[:-test_count]
    run_slda(train_reviews, test_reviews, K=10, train_iters=10)
