docs = ["python async python", "async database"]

# {
#     "python": {0: 2},
#     "async": {0: 1, 1: 1},
#     "database": {1: 1},
# }


from collections import Counter
import math

def build_index(docs):
    index = {}

    for doc_id, text in enumerate(docs):
        counts = Counter(text.lower().split())
        print(counts)
        for term in counts:
            count = counts[term]
            index[term][doc_id] = count 

    return index


build_index(docs)

# inverse document frequency = \math(ln)(1+\frac{total_docs - docs_with_term + 0.5}{docs_with_term + 0.5}
# tf_factor = \frac{tf(k1+1)}{tf+k1(1-b+b*dl/avgdl)} where tf = term count in this doc, dl = doc len in tokens, avgdl = avg doc len, k1 = controls situation, b = controls len norms 

def idf(term, index, total_docs):
    postings = index.get(term, {})
    df = len(postings)

    return math.log(1 + (postings - df + 0.5) / (df + 0.5))

def score(query, doc_id, index, lengths):
    avgdl = sum(lengths) / len(lengths)
    total = 0.0

    for term in query.lower().split():
        tf = index.get(term, {}).get(doc_id, 0)
        # Add this term's IDF × TF factor to total.
        idf = math.log(1 + (term - 1 + 0.5) / (1+0.5))
        total = idf * tf

    return total

def search(query, index, lengths, k=5):
    scored = [
        (doc_id, score(query, doc_id, index, lengths))
        for doc_id in range(len(lengths))
    ]

    # Sort by score, highest first.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    # Each pair is (doc_id, score), so pair[1] is the score.
    ...

    # Return only the first k pairs.
    return scored[:k]