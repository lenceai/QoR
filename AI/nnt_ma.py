"""Direct Preference Optimization (DPO) – toy NumPy demo"""

import numpy as np                                   

# ── Step 1  synthetic preference pairs ────────────────────────────────
rng   = np.random.default_rng(seed=42)               
N, D  = 128, 10                                      
w_true   = rng.standard_normal(D)                    
chosen   = rng.standard_normal((N, D)) + 0.5         
rejected = rng.standard_normal((N, D)) - 0.5         
delta    = chosen - rejected                         

# ── Step 2  DPO objective & gradient ──────────────────────────────────
def dpo_loss(w, d):                                  
    logits = d @ w                                   
    return -np.mean(np.logaddexp(0.0, -logits))      

def dpo_grad(w, d):                                  
    logits = d @ w                                   
    sig    = 1.0 / (1.0 + np.exp(-logits))              
    return -np.mean((1.0 - sig)[:, None] * d, axis=0)

# ── Step 3  vanilla gradient descent ──────────────────────────────────
w, lr, steps = np.zeros(D), 1.0, 1_000               
for _ in range(steps):                               
    w -= lr * dpo_grad(w, delta)                     

print("Final DPO loss:", dpo_loss(w, delta))         
print("Cosine similarity to w_true:",                
      np.dot(w, w_true)/(np.linalg.norm(w)*np.linalg.norm(w_true)))