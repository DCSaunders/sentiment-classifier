from __future__ import division

from collections import defaultdict
import numpy as np
import tokenizer
import logging
from scipy.stats import norm

class Topic(object):
    def __init__(self):
        # Store counts of words given topic, across all documents
        self.word_counts = defaultdict(int)
        
def initialise(docs, topics, vocab, K):
    topic_word_assign = np.zeros(K)
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
    return words_given_topics, topic_word_assign

def rating_to_regr(rating, reverse=False):
    if not reverse:
        # regression targets are 0 and 1
        return (rating + 1) / 2
    else:
        return (rating - 0.5) * 2
           
def train(docs, topics, train_iters, vocab_size,
          topic_word_assign, words_given_topics, eta,
          alpha, gamma):
    K = len(topics)
    sample_reg_every = 100
    logging.info('Sampling eta every {} docs'.format(sample_reg_every))
    labels = np.array([rating_to_regr(doc.rating) for doc in docs])
    
    for i in range(0, train_iters):
        logging.info('Iteration {}'.format(i))
        for doc_index, doc in enumerate(docs):
            for index, word in enumerate(doc.text_no_stopwords):
                old_topic = doc.topic_words[index]    
                topics[old_topic].word_counts[word] -= 1
                topic_word_assign[old_topic] -= 1
                doc.topic_counts[old_topic] -= 1
                words_given_topics[word][old_topic] -= 1
            
                z_bar = np.zeros([K, K]) + doc.topic_counts + np.identity(K)
                z_bar = z_bar / z_bar.sum(1)
                y_diff = labels[doc_index] - np.dot(z_bar, eta)

                distrib = ((alpha + doc.topic_counts)
                           * (gamma + words_given_topics[word])
                           * np.exp(-0.5 * y_diff ** 2)
                           / (vocab_size * gamma + topic_word_assign))
                new_topic = sample_discrete(distrib)
                doc.topic_words[index] = new_topic
                doc.topic_counts[new_topic] += 1
                topics[new_topic].word_counts[word] += 1
                topic_word_assign[new_topic] += 1
                words_given_topics[word][new_topic] += 1
            if doc_index % sample_reg_every == 0 or doc_index == len(docs) - 1:
                # resample eta after topic burn-in
                doc_topics = np.array([doc.topic_counts for doc in docs])
                topic_probs = doc_topics / doc_topics.sum(1)[:, np.newaxis] 
                eta = np.linalg.solve(np.dot(topic_probs.T, topic_probs),
                                      np.dot(topic_probs.T, labels))
    return eta, topic_word_assign
                
def sample_discrete(distribution):
    r = sum(distribution) * np.random.uniform()
    total = 0
    for choice, p in enumerate(distribution):
        total += p
        if total >= r:
            return choice

def run_slda(train_docs, test_docs, K, train_iters, alpha, gamma):
    top_words = 10
    topics = []
    test_topics = []
    eta_scale = 5
    eta = eta_scale * np.random.randn(K) # regression coefficients for topics
    vocab = set()
    for review in train_docs:
        vocab = vocab.union(review.text_no_stopwords)
    vocab_size = len(vocab)
    logging.info(
        'sLDA with vocab size {}, {} training iterations, {} topics, eta linspace init, alpha {}, gamma {}, eta scale {}'.format(
        vocab_size, train_iters, K, alpha, gamma, eta_scale))
    for t in range(0, K):
        topics.append(Topic())
        test_topics.append(Topic())
    words_given_topics, topic_word_assign = initialise(
        train_docs, topics, vocab, K)
    eta, topic_word_assign = train(
        train_docs, topics, train_iters, vocab_size, topic_word_assign,
        words_given_topics, eta, alpha, gamma)
    logging.info('sLDA eta is {}'.format(eta))
    initialise(test_docs, test_topics, vocab, K)
    ymu_accuracies = {}
    ymu_probs = {}
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
        y_mu = np.dot(topic_probs, eta)
        ymu_est = -1 if y_mu < 0.5 else 1
        ymu_probs[doc] = norm.cdf(y_mu - 0.5)
        if ymu_est == doc.rating:
            ymu_accuracies[doc] = 1
        else:
            ymu_accuracies[doc] = 0
    for index, topic in enumerate(topics):
        top_topic_words = sorted(topic.word_counts,
                                 key=lambda x: topic.word_counts[x],
                                 reverse=True)[:top_words]
        logging.info('{}: {}'.format(index, ' '.join(top_topic_words)))
    return ymu_accuracies, ymu_probs

        
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
