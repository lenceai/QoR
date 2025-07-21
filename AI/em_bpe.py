from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=100, min_frequency=2):
        # Initialize the tokenizer with target vocab size and minimum pair frequency
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges = []       # list to store merge rules as tuples (token_a, token_b)
        self.token2id = {}     # dictionary mapping token string -> token ID
        self.id2token = {}     # dictionary mapping token ID -> token string

    def train(self, corpus):
        # Training: learn BPE merges from the given corpus text.
        # corpus can be a single string or a list of strings.
        if isinstance(corpus, str):
            words = corpus.split()
        else:
            words = []
            for text in corpus:
                words.extend(text.split())
        # Append '_' marker to each word to mark end-of-word
        words = [word + '_' for word in words]
        # Count frequency of each distinct word in the corpus
        word_freq = Counter(words)
        # Initialize base vocabulary with all unique characters (including '_')
        base_tokens = set()
        for word in word_freq:
            base_tokens.update(list(word))
        base_tokens = sorted(base_tokens)  # sort for consistency (optional)
        # Assign an ID to each base token (character)
        self.token2id = {token: idx for idx, token in enumerate(base_tokens)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        # Represent each word as a list of character tokens (with '_')
        word_tokens = {word: list(word) for word in word_freq}
        # Learn merge rules until vocab size reached or no frequent pair meets threshold
        while len(self.token2id) < self.vocab_size:
            # Count frequency of each adjacent token pair across all words
            pair_counts = Counter()
            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                # count pairs in this word's token sequence
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i+1])
                    pair_counts[pair] += freq
            if not pair_counts:
                break  # no pairs to merge (shouldn't really happen unless corpus is empty)
            # Find the most frequent pair
            (token_a, token_b), pair_freq = pair_counts.most_common(1)[0]
            if pair_freq < self.min_frequency:
                break  # stop if no pair is frequent enough
            # Merge this pair into a new token
            new_token = token_a + token_b
            # Add new token to vocab with the next available ID
            new_id = len(self.token2id)
            self.token2id[new_token] = new_id
            self.id2token[new_id] = new_token
            # Record the merge rule
            self.merges.append((token_a, token_b))
            # Update word token sequences: replace occurrences of the pair with the new token
            for word, tokens in word_tokens.items():
                i = 0
                new_tokens = []
                while i < len(tokens) - 1:
                    if tokens[i] == token_a and tokens[i+1] == token_b:
                        # Merge token_a and token_b into new_token
                        new_tokens.append(new_token)
                        i += 2  # skip over the merged pair
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                # Don't forget the last token if it wasn't part of a merge
                if i < len(tokens):
                    new_tokens.append(tokens[i])
                # Update the word's token list
                word_tokens[word] = new_tokens
            # Continue to next merge iteration
        # Training complete. We have our merges and vocab.

    def encode(self, text):
        # Convert a text string into a list of token IDs using learned merges.
        tokens_ids = []
        # Split input text into words and encode each word
        for word in text.split():
            # Start with characters + '_' for the word
            tokens = list(word) + ['_']
            # Apply each merge rule in order to the token list
            for token_a, token_b in self.merges:
                i = 0
                merged_tokens = []
                while i < len(tokens) - 1:
                    if tokens[i] == token_a and tokens[i+1] == token_b:
                        merged_tokens.append(token_a + token_b)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1
                if i < len(tokens):
                    merged_tokens.append(tokens[i])
                tokens = merged_tokens
            # Convert tokens to IDs
            for token in tokens:
                # .get(token) will fetch the ID; all tokens should exist from training
                tokens_ids.append(self.token2id.get(token, None))
        return tokens_ids

    def decode(self, token_ids):
        # Convert a list of token IDs back into the original text string.
        tokens = [self.id2token[token_id] for token_id in token_ids]
        text = ""
        for token in tokens:
            if token.endswith('_'):
                # Remove the end-of-word marker and add a space
                text += token[:-1] + " "
            else:
                # Token without marker (should be punctuation or part of word that isn't ending)
                text += token
        return text.strip()  # strip any trailing space

# Sample corpus for training
corpus = [
    "low", "lower", "newest", "widest",
    "low", "low", "low", "low",  # Repetition to build frequency
    "newest", "newest", "newest", "newest"
]

# Initialize tokenizer with small vocab size
tokenizer = BPETokenizer(vocab_size=50, min_frequency=2)
tokenizer.train(corpus)

# Print learned vocabulary
print("Vocabulary:", tokenizer.token2id)
print("Merge rules:", tokenizer.merges)

# Encode a new word
encoded = tokenizer.encode("lowest")
print("Encoded 'lowest':", encoded)

# Decode the tokens back
decoded = tokenizer.decode(encoded)
print("Decoded text:", decoded)