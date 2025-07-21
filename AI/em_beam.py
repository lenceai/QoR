import math
from typing import List, Tuple
import random

EOS_TOKEN = "<EOS>"

def beam_search(start_token: str,
                get_next_token_probs,
                beam_width: int = 3,
                max_length: int = 10) -> List[str]:
    # Each beam entry is a tuple: (sequence, total log-prob)
    beam = [([start_token], 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, seq_logprob in beam:
            if seq[-1] == EOS_TOKEN:
                # Already ended — carry forward unchanged
                all_candidates.append((seq, seq_logprob))
                continue

            # Get top next-token candidates and their log-probs
            next_tokens = get_next_token_probs(seq)

            for token, log_prob in next_tokens:
                new_seq = seq + [token]
                new_logprob = seq_logprob + log_prob
                all_candidates.append((new_seq, new_logprob))

        # Keep the top-k sequences with highest log-prob
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beam = all_candidates[:beam_width]

        # Stop early if all beams have ended
        if all(seq[-1] == EOS_TOKEN for seq, _ in beam):
            break

    best_sequence, _ = beam[0]
    return best_sequence

def get_next_token_probs(seq: List[str]) -> List[Tuple[str, float]]:
    vocab = ["hello", "world", "foo", "bar", "<EOS>"]
    return [(token, math.log(random.uniform(0.1, 1.0))) for token in vocab]

output = beam_search("<START>", get_next_token_probs, beam_width=3, max_length=5)
print("Generated sequence:", " ".join(output))