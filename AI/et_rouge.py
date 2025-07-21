from collections import Counter

def compute_rouge_1(candidate, reference):
    # Tokenize the sentences
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    # Count unigrams
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)

    # Calculate overlapping unigrams
    overlapping_unigrams = sum(min(candidate_counts[word], reference_counts[word]) for word in candidate_counts)

    # Calculate recall
    recall = overlapping_unigrams / max(1, len(reference_tokens))

    # Calculate precision
    precision = overlapping_unigrams / max(1, len(candidate_tokens))

    # Calculate F1-score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }

# Example usage
candidate = "the cat sat on the mat"
reference = "the cat is sitting on the mat"

scores = compute_rouge_1(candidate, reference)
print(f"ROUGE-1 Precision: {scores['precision']:.2f}")
print(f"ROUGE-1 Recall: {scores['recall']:.2f}")
print(f"ROUGE-1 F1 Score: {scores['f1']:.2f}")