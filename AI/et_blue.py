import math
from collections import Counter

def compute_bleu(candidate, references, max_n=4):

    # Tokenize candidate and references
    candidate_tokens = candidate.split()
    references_tokens = [ref.split() for ref in references]

    precisions = []
    for n in range(1, max_n+1):
        # Get n-grams for candidate
        candidate_ngrams = Counter(tuple(candidate_tokens[i:i+n]) for i in range(len(candidate_tokens)-n+1))
        max_ref_ngrams = Counter()

        # Get max reference n-grams counts
        for ref in references_tokens:
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            for ngram in ref_ngrams:
                max_ref_ngrams[ngram] = max(max_ref_ngrams.get(ngram,0), ref_ngrams[ngram])

        # Clip candidate n-gram counts by reference max counts
        clipped_counts = {ngram: min(count, max_ref_ngrams.get(ngram,0)) for ngram, count in candidate_ngrams.items()}

        precision = sum(clipped_counts.values()) / max(1, sum(candidate_ngrams.values()))
        precisions.append(precision)

    # Geometric mean of precisions
    if min(precisions) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    else:
        geo_mean = 0

    # Brevity penalty
    ref_lens = [len(ref) for ref in references_tokens]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - len(candidate_tokens)), ref_len))
    bp = math.exp(1 - closest_ref_len / len(candidate_tokens)) if len(candidate_tokens) < closest_ref_len else 1

    return bp * geo_mean

# Example
candidate = "the cat is on mat"
references = ["the cat is on the mat", "there is a cat on the mat"]

print(f"BLEU score: {compute_bleu(candidate, references):.4f}")