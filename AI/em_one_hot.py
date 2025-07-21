# One-hot encoding in simple Python
vocab = ['cat', 'kitten', 'bank', 'Java']
target = 'Java'
index = vocab.index(target)
one_hot = [1 if i == index else 0 for i in range(len(vocab))]

print(one_hot)  # Output: [0, 0, 0, 1]

import numpy as np
from collections import defaultdict
import random

# ---------------------
# 1. Prepare the corpus
# ---------------------
corpus = [
    "king is a strong man",
    "queen is a wise woman",
    "man and woman are humans",
    "prince is a young king",
    "princess is a young queen"
]

# Tokenize each sentence into a list of lowercase words
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Build word frequency dictionary
word_counts = defaultdict(int)
for sentence in tokenized_corpus:
    for word in sentence:
        word_counts[word] += 1

# Create vocabulary and lookup tables
vocab = list(word_counts.keys())
word2idx = {w: idx for idx, w in enumerate(vocab)}    # word to index mapping
idx2word = {idx: w for w, idx in word2idx.items()}    # index to word mapping
vocab_size = len(vocab)

# ----------------------------
# 2. Generate skip-gram pairs
# ----------------------------
def generate_training_data(tokenized_corpus, window_size=2):
    """
    For each word in a sentence, return (center, context) pairs within the given window.
    """
    pairs = []
    for sentence in tokenized_corpus:
        for idx, center_word in enumerate(sentence):
            for j in range(-window_size, window_size + 1):
                context_idx = idx + j
                if j != 0 and 0 <= context_idx < len(sentence):
                    context_word = sentence[context_idx]
                    pairs.append((center_word, context_word))
    return pairs

# Create training data
training_data = generate_training_data(tokenized_corpus)

# ----------------------------
# 3. Initialize model weights
# ----------------------------
embedding_dim = 10   # Size of word embedding vector

# Weight matrix from input to hidden layer
W1 = np.random.rand(vocab_size, embedding_dim)

# Weight matrix from hidden to output layer
W2 = np.random.rand(embedding_dim, vocab_size)

# -----------------------
# 4. Softmax and training
# -----------------------
def softmax(x):
    """
    Stable softmax function to convert raw scores into probabilities.
    """
    e_x = np.exp(x - np.max(x))  # stability trick
    return e_x / e_x.sum(axis=0)

def train(epochs=1000, learning_rate=0.01):
    """
    Train the model using naive softmax cross-entropy loss and SGD.
    """
    global W1, W2
    for epoch in range(epochs):
        loss = 0
        for center, context in training_data:
            # Create one-hot vector for center word
            x = np.zeros(vocab_size)
            x[word2idx[center]] = 1

            # Forward pass
            h = np.dot(W1.T, x)          # hidden layer (center word projection)
            u = np.dot(W2.T, h)          # output layer scores
            y_pred = softmax(u)          # predicted probability distribution

            # True distribution (one-hot for context word)
            y_true = np.zeros(vocab_size)
            y_true[word2idx[context]] = 1

            # Compute error
            e = y_pred - y_true

            # Backpropagate errors to weights
            dW2 = np.outer(h, e)
            dW1 = np.outer(x, np.dot(W2, e))

            # Update weights
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2

            # Accumulate loss
            loss += -np.log(y_pred[word2idx[context]])
        
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Train the model
train()

# -------------------
# 5. Query word vector
# -------------------
def get_word_vector(word):
    """
    Return the learned embedding vector for a given word.
    """
    idx = word2idx[word]
    return W1[idx]

# Show learned vectors for "king" and "queen"
print("\nVector for 'king':")
print(get_word_vector("king"))

print("\nVector for 'queen':")
print(get_word_vector("queen"))