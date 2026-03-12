import torch
import math

# We will use simple, reproducible numbers and 100% manual calculation.
torch.manual_seed(42)

# ==============================================================================
# 0. SETUP THE DATA
# ==============================================================================
print("\n--- 0. SETUP THE DATA ---")
B, T, C = 1, 3, 4

# Input x (3 words, 4 features each)
x = torch.randn(B, T, C)
print(f"Input x Shape: {list(x.size())}")
print(f"Input Data x[0]:\n{x[0]}")

# ==============================================================================
# 1. LAYER NORMALIZATION 1 (Manual math, no F.layer_norm)
# ==============================================================================
print("\n--- 1. LAYER NORMALIZATION 1 ---")
# LayerNorm calculates the mean and variance across the feature dimension (dim=2)
# Formula: y = (x - mean) / sqrt(variance + epsilon)

epsilon = 1e-5
# 1. Calculate Mean (sum of 4 features / 4)
mean = x.mean(dim=2, keepdim=True) 

# 2. Calculate Variance (average squared distance from mean)
variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)

# 3. Normalize
norm_x = (x - mean) / torch.sqrt(variance + epsilon)

print(f"Mean across features for Word 1: {mean[0, 0].item():.4f}")
print(f"Variance for Word 1: {variance[0, 0].item():.4f}")
print(f"Normalized Data (norm_x[0]):\n{norm_x[0]}")


# ==============================================================================
# 2. ATTENTION: Q, K, V PROJECTION (Manual MatMul)
# ==============================================================================
print("\n--- 2. Q, K, V PROJECTION ---")

# Instead of one giant weight matrix, let's explicitly make 3 separate ones
# to understand what the linear layer is actually doing underneath.
W_q = torch.randn(C, C) # A 4x4 weight matrix to turn (3,4) into queries
W_k = torch.randn(C, C) # A 4x4 matrix for keys
W_v = torch.randn(C, C) # A 4x4 matrix for values

# Manual Matrix Multiplication
Q = norm_x @ W_q
K = norm_x @ W_k
V = norm_x @ W_v

print(f"Q Shape: {list(Q.size())}")
print(f"K Shape: {list(K.size())}")


# ==============================================================================
# 3. SPLITTING HEADS (Manual Indexing instead of fancy transposes)
# ==============================================================================
print("\n--- 3. SPLITTING INTO 2 HEADS ---")
# We have 4 features. Head 1 gets features 0,1. Head 2 gets features 2,3.
# We will do this manually by slicing the tensors!

# Head 1 Context:
Q_h1 = Q[:, :, 0:2] # Take first 2 features
K_h1 = K[:, :, 0:2]
V_h1 = V[:, :, 0:2]

# Head 2 Context:
Q_h2 = Q[:, :, 2:4] # Take last 2 features
K_h2 = K[:, :, 2:4]
V_h2 = V[:, :, 2:4]

print(f"Head 1 Q Shape: {list(Q_h1.size())}")
print(f"Head 2 Q Shape: {list(Q_h2.size())}")


# ==============================================================================
# 4. ATTENTION CALCULATION (Manual math for Head 1)
# ==============================================================================
print("\n--- 4. ATTENTION CALCULATION (Showing Head 1) ---")

# A. Dot Product (How much does word i care about word j?)
# We transpose K so we can multiply a (3,2) matrix by a (2,3) matrix -> (3,3)
scores_h1 = Q_h1 @ K_h1.transpose(1, 2) 

# B. Scale by sqrt(d_k)
d_k = 2 # size of head
scores_h1 = scores_h1 / math.sqrt(d_k)

# C. Causal Mask (Manual)
# Force future word scores to -infinity so they become 0 after softmax
scores_h1[0, 0, 1] = float('-inf') # Word 1 can't see Word 2
scores_h1[0, 0, 2] = float('-inf') # Word 1 can't see Word 3
scores_h1[0, 1, 2] = float('-inf') # Word 2 can't see Word 3

# D. Softmax (Manual math without F.softmax)
# Formula: e^x / sum(e^x)
exp_scores_h1 = torch.exp(scores_h1)
sum_exp_h1 = exp_scores_h1.sum(dim=-1, keepdim=True)
probs_h1 = exp_scores_h1 / sum_exp_h1

print(f"Head 1 Attention Probabilities (3x3):\n{probs_h1[0]}")

# E. Weighted Sum of Values
output_h1 = probs_h1 @ V_h1

# Let's quickly do Head 2 identically for the sake of the next step
scores_h2 = (Q_h2 @ K_h2.transpose(1, 2)) / math.sqrt(d_k)
scores_h2[0, 0, 1], scores_h2[0, 0, 2], scores_h2[0, 1, 2] = float('-inf'), float('-inf'), float('-inf')
probs_h2 = torch.exp(scores_h2) / torch.exp(scores_h2).sum(dim=-1, keepdim=True)
output_h2 = probs_h2 @ V_h2


# ==============================================================================
# 5. CONCATENATION AND FINAL PROJECTION
# ==============================================================================
print("\n--- 5. RE-ASSEMBLING THE HEADS ---")
# Manually glue Head 1 (features 0,1) and Head 2 (features 2,3) back together
concat_output = torch.cat((output_h1, output_h2), dim=2)
print(f"Concatenated Output Shape: {list(concat_output.size())}")

# Final linear projection (mixing the heads context via matrix multiplication)
W_proj_attn = torch.randn(C, C)
attn_final = concat_output @ W_proj_attn

# ==============================================================================
# 6. RESIDUAL CONNECTION 1
# ==============================================================================
print("\n--- 6. RESIDUAL CONNECTION 1 ---")
x = x + attn_final # The original word + its new context
print(f"x Shape after Residual 1: {list(x.size())}")


# ==============================================================================
# 7. LAYER NORMALIZATION 2 (Manual)
# ==============================================================================
print("\n--- 7. LAYER NORMALIZATION 2 ---")
mean = x.mean(dim=2, keepdim=True)
variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)
norm_x = (x - mean) / torch.sqrt(variance + epsilon)

# ==============================================================================
# 8-10. MLP (Manual Math, No nn.Linear)
# ==============================================================================
print("\n--- 8. MULTI-LAYER PERCEPTRON (Manual Math) ---")

# A. Expansion (Multiply by a 4x16 weight matrix)
W_expand = torch.randn(C, 4 * C) # Expand from 4 to 16 dimensions
mlp_hidden = norm_x @ W_expand
print(f"Expanded Shape: {list(mlp_hidden.size())}")

# B. GELU Activation
# We use standard PyTorch GELU here as the manual polynomial approximation is very long
mlp_activated = torch.nn.functional.gelu(mlp_hidden)

# C. Compression (Multiply by a 16x4 weight matrix)
W_compress = torch.randn(4 * C, C) # Compress back to 4 dimensions
mlp_final = mlp_activated @ W_compress
print(f"Compressed Shape: {list(mlp_final.size())}")

# ==============================================================================
# 11. RESIDUAL CONNECTION 2
# ==============================================================================
print("\n--- 11. RESIDUAL CONNECTION 2 ---")
x = x + mlp_final

print(f"FINAL OUTPUT SHAPE OF THE BLOCK: {list(x.size())}")
print(f"Final Output Data x[0]:\n{x[0]}")
