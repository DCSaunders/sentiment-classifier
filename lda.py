from collections import defaultdict

doc_a = "Brocolli is good to eat My brother likes to eat good brocolli, but not my mother"
doc_b = "My mother spends a lot of time driving my brother around to baseball practice"
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure"
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better"
doc_e = "Health professionals say that brocolli is good for your health"

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
tokens = defaultdict(int)
for doc in doc_set:
    doc = doc.lower().split()
    for tok in doc:
        tokens[tok] += 1
    
