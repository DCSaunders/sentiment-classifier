from __future__ import division

from collections import defaultdict
import numpy as np
import tokenizer
import logging

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

           
def train(docs, topics, train_iters, vocab_size,
          topic_word_assign, words_given_topics, alpha, gamma):
    K = len(topics)
    for i in range(0, train_iters):
        logging.info('Iteration {}'.format(i))
        for doc in docs:
            for index, word in enumerate(doc.text_no_stopwords):
                old_topic = doc.topic_words[index]    
                topics[old_topic].word_counts[word] -= 1
                topic_word_assign[old_topic] -= 1
                doc.topic_counts[old_topic] -= 1
                words_given_topics[word][old_topic] -= 1
                # discrete distribution over topics
                distrib = ((alpha + doc.topic_counts)
                           * (gamma + words_given_topics[word])
                           / (vocab_size * gamma + topic_word_assign))
                new_topic = sample_discrete(distrib)
                doc.topic_words[index] = new_topic
                doc.topic_counts[new_topic] += 1
                topics[new_topic].word_counts[word] += 1
                topic_word_assign[new_topic] += 1
                words_given_topics[word][new_topic] += 1
    return topic_word_assign

def sample_discrete(distribution):
    r = sum(distribution) * np.random.uniform()
    total = 0
    for choice, p in enumerate(distribution):
        total += p
        if total >= r:
            return choice

def run_lda(train_docs, test_docs, K, train_iters=100, alpha=0.1, gamma=0.1):
    top_words = 10
    topics = []
    test_topics = []
    vocab = set()
    for review in train_docs:
        vocab = vocab.union(review.text_no_stopwords)
    vocab_size = len(vocab)
    logging.info('LDA with vocab size {}, {} training iterations, {} topics, alpha={}, gamma={}'.format(vocab_size, train_iters, K, alpha, gamma))
    for t in range(0, K):
        topics.append(Topic())
        test_topics.append(Topic())
  
    words_given_topics, topic_word_assign = initialise(
        train_docs, topics, vocab, K)
    topic_word_assign = train(train_docs, topics, train_iters,
                              vocab_size, topic_word_assign,
                              words_given_topics, alpha, gamma)
    initialise(test_docs, test_topics, vocab, K)
    
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
    run_lda(train_reviews, test_reviews, K=10, train_iters=10)
