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
        for word in doc.text_no_stopwords:
            t = np.random.randint(0, K)
            doc.topic_words.append(t)
            doc.topic_counts[t] += 1
            topics[t].word_counts[word] += 1

def estimate_probs(docs, topics):
    for topic in topics:
        total = sum(topic.word_counts.values())
        for word, count in topic.word_counts.items():
            topic.word_probs[word] = count / total
    for doc in docs:
        total = len(doc.topic_words)
        for t in range(0, len(topics)):
            doc.topic_probs[t] = doc.topic_counts[t] / total
            
def train(docs, topics, train_iters):
    for _ in range(0, train_iters):
        estimate_probs(docs, topics)
        for doc in docs:
            for index, word in enumerate(doc.text):
                old_topic = doc.topic_assignments[index]
                p_w_t = np.array(
                    [t.word_probs[word] for t in topics])
                new_topic = np.argmax(p_w_t*doc.topic_probs)

                doc.topic_assignments[index] = new_topic
                doc.topic_counts[old_topic] -= 1
                doc.topic_counts[new_topic] += 1
                topics[old_topic].word_counts[word] -= 1
                topics[new_topic].word_counts[word] += 1

        
np.random.seed(1234)
K = 4
doc_count = 5
train_iters = 1000
top_words = 3
reviews = []
topics = []
alpha = 0.1 # dirichlet parameter over topics (per review)
gamma = 0.1 # dirichlet parameter over words

tokenizer.tokenize_files('data/POS', reviews, set())
vocab = set()
reviews = reviews[0:doc_count]
for review in reviews:
    vocab = vocab.union(review.text_no_stopwords)
vocab_size = len(vocab)

for t in range(0, K):
    topics.append(Topic())
    
initialise(reviews, topics, K)
train(docs, topics, train_iters)
for doc in docs:
    print doc.text, np.argmax(doc.topic_probs)
for topic in topics:
    words = topic.word_counts.keys()
    counts = np.array([topic.word_counts[w] for w in words])
    top = np.argpartition(counts, -top_words)[-top_words:]
    top = top[np.argsort(counts[top])]
    print [words[ind] for ind in top]
