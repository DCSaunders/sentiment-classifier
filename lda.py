from __future__ import division

from collections import defaultdict
import numpy as np
import tokenizer

class Topic(object):
    def __init__(self):
        # Store counts/probs of words given topic, across all documents
        self.word_counts = defaultdict(int)
        self.word_probs = defaultdict(float)
        
def initialise(docs, topics, K):
    for doc in docs:
        doc.topic_counts, doc.topic_probs = np.zeros(K), np.zeros(K)
        for word in doc.text_no_stopwords:
            t = np.random.randint(0, K)
            doc.topic_words.append(t)
            doc.topic_counts[t] += 1
            topics[t].word_counts[word] += 1

def estimate_probs(docs, topics, vocab):
    for topic in topics:
        total = sum(topic.word_counts.values())
        for word, count in topic.word_counts.items():
            topic.word_probs[word] = count / total
    for doc in docs:
        total = len(doc.topic_words)
        doc.topic_probs = doc.topic_counts / total
    words_given_topics = {w: np.array([t.word_probs[w] for t in topics])
                          for w in vocab}
    return words_given_topics
            
def train(docs, topics, train_iters, vocab):
    for i in range(0, train_iters):
        print 'Iteration {}'.format(i)
        words_given_topics = estimate_probs(docs, topics, vocab)
        for doc in docs:
            for index, word in enumerate(doc.text_no_stopwords):
                old_topic = doc.topic_words[index]
                p_w_t = words_given_topics[word]
                new_topic = np.argmax(p_w_t*doc.topic_probs)
                doc.topic_words[index] = new_topic
                doc.topic_counts[old_topic] -= 1
                doc.topic_counts[new_topic] += 1
                topics[old_topic].word_counts[word] -= 1
                topics[new_topic].word_counts[word] += 1

        
np.random.seed(1234)
K = 10
doc_count = 100
train_iters = 1000
top_words = 20
reviews = []
topics = []
alpha = 0.1 # dirichlet parameter over topics (per review)
gamma = 0.1 # dirichlet parameter over words

tokenizer.tokenize_files('20news-bydate-train/sci.crypt', reviews, set())
vocab = set()
reviews = reviews[0:doc_count]
for review in reviews:
    vocab = vocab.union(review.text_no_stopwords)
vocab_size = len(vocab)
print vocab_size

for t in range(0, K):
    topics.append(Topic())
    
initialise(reviews, topics, K)
train(reviews, topics, train_iters, vocab)
for index, review in enumerate(reviews):
    print index, np.argmax(review.topic_probs)
for index, topic in enumerate(topics):
    words = topic.word_counts.keys()
    counts = np.array([topic.word_counts[w] for w in words])
    top = np.argpartition(counts, -top_words)[-top_words:]
    top = top[np.argsort(counts[top])]
    print index, [words[ind] for ind in top]
