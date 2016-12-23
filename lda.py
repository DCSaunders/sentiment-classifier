from __future__ import division

from collections import defaultdict
import numpy as np
import tokenizer

class Topic(object):
    def __init__(self):
        # Store counts/probs of words given topic, across all documents
        self.word_counts = defaultdict(int)
        
def initialise(docs, topics, topic_word_assignments, K):
    for doc in docs:
        doc.topic_counts, doc.topic_probs = np.zeros(K), np.zeros(K)
        for word in doc.text_no_stopwords:
            t = np.random.randint(0, K)
            doc.topic_words.append(t)
            doc.topic_counts[t] += 1
            topics[t].word_counts[word] += 1
    for topic_idx, topic in enumerate(topics):
        topic_word_assignments[topic_idx] = sum(topic.word_counts.values())
    words_given_topics = defaultdict(list)
    for w in vocab:
        words_given_topics[w] = np.array([t.word_counts[w] for t in topics])
    return words_given_topics

           
def train(docs, topics, train_iters, vocab_size,
          topic_word_assignments, words_given_topics):
    for i in range(0, train_iters):
        print 'Iteration {}'.format(i)
        for doc in docs:
            for index, word in enumerate(doc.text_no_stopwords):
                '''
   skd(:,d) is the vector of topic counts for that doc (1*K)
   swk(w,:) is the vector of topic counts for that word (over all docs) (1*K) 
   W is the overall vocabulary size (scalar)
   sk is the vector of total counts of words assigned to each topic (1*K)
   Probability vectors probably not needed, just count vectors.
   Also code deals with each instance of word-assigned-to-topic at a time - 
   first remove from relevant count matrices, then do calculation below
        b = (alpha + skd(:,d)) .* (gamma + swk(w,:)') ./ (W*gamma + sk);
        kk = sampDiscrete(b);     % Gibbs sample new topic assignment
                '''
                old_topic = doc.topic_words[index]    
                topics[old_topic].word_counts[word] -= 1
                topic_word_assignments[old_topic] -= 1
                doc.topic_counts[old_topic] -= 1
                # discrete distribution over topics
                distrib = ((alpha + doc.topic_counts)
                           * (gamma + words_given_topics[word])
                           / (vocab_size * gamma + topic_word_assignments))
                new_topic = sample_discrete(distrib)
                doc.topic_words[index] = new_topic
                
                doc.topic_counts[new_topic] += 1
                topics[new_topic].word_counts[word] += 1
                topic_word_assignments[new_topic] += 1
        

def sample_discrete(distribution):
    # sample from discrete count distribution
    r = sum(distribution) * np.random.uniform()
    total = 0
    for choice, p in enumerate(distribution):
        total += p
        if total >= r:
            return choice
                
np.random.seed(1234)
K = 5
doc_count = 100
train_iters = 100
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
topic_word_assignments = np.zeros(K)
    
words_given_topics = initialise(reviews, topics, topic_word_assignments, K)
train(reviews, topics, train_iters,
      vocab_size, topic_word_assignments, words_given_topics)
for index, review in enumerate(reviews):
    print index, np.argmax(review.topic_probs)
for index, topic in enumerate(topics):
    words = topic.word_counts.keys()
    counts = np.array([topic.word_counts[w] for w in words])
    top = np.argpartition(counts, -top_words)[-top_words:]
    top = top[np.argsort(counts[top])]
    print index, [words[ind] for ind in top]
