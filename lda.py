from __future__ import division

from collections import defaultdict
import numpy as np

class Doc(object):
    def __init__(self, text, K):
        # Store word topic assignments and counts/probs of topic given this document
        self.text = text
        self.topic_assignments = []
        self.topic_counts = np.zeros(K)
        self.topic_probs = np.zeros(K)

class Topic(object):
    def __init__(self):
        # Store counts/probs of words given topic, across all documents
        self.word_counts = defaultdict(int)
        self.word_probs = defaultdict(float)
        
def initialise(docs, topics, K):
    for doc in docs:
        for word in doc.text:
            t = np.random.randint(0, K)
            doc.topic_assignments.append(t)
            doc.topic_counts[t] += 1
            topics[t].word_counts[word] += 1

def estimate_probs(docs, topics):
    for topic in topics:
        total = sum(topic.word_counts.values())
        for word, count in topic.word_counts.items():
            topic.word_probs[word] = count / total
    for doc in docs:
        total = sum(doc.topic_counts)
        for t in range(0, len(doc.topic_probs)):
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

            
text = ["brocolli is good to eat my brother likes to eat good brocolli, but not my mother",
 "my mother spends a lot of time driving my brother around to baseball practice",
 "some health experts suggest that driving may cause increased tension and blood pressure",
 "i often feel pressure to perform well at school, but my mother never seems to drive my brother to do better",
"health professionals say that brocolli is good for your health"]
np.random.seed(1234)
K = 4
train_iters = 1000
docs = []
topics = []

for string in text:
    docs.append(Doc(string.split(), K))
for t in range(0, K):
    topics.append(Topic())
    
initialise(docs, topics, K)
train(docs, topics, train_iters)
for doc in docs:
    print doc.text, np.argmax(doc.topic_probs)
