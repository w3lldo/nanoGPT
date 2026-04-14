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
# OUTPUT: Input x Shape: [1, 3, 4]

print(f"Input Data x[0]:\n{x[0]}")
# OUTPUT: Input Data x[0]:
# tensor([[ 0.3367,  0.1288,  0.2345,  0.2303],
#         [-1.1229, -0.1863,  2.2082, -0.6380],
#         [ 0.4617,  0.2674,  0.5349,  0.8094]])

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
# OUTPUT: Mean across features for Word 1: 0.2326

print(f"Variance for Word 1: {variance[0, 0].item():.4f}")
# OUTPUT: Variance for Word 1: 0.0054

print(f"Normalized Data (norm_x[0]):\n{norm_x[0]}")
# OUTPUT: Normalized Data (norm_x[0]):
# tensor([[ 1.4150, -1.4102,  0.0257, -0.0305],
#         [-0.9276, -0.1964,  1.6731, -0.5491],
#         [-0.2914, -1.2908,  0.0853,  1.4969]])


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
print(f"Q Data:\n{Q[0]}")
# OUTPUT: Q Shape: [1, 3, 4]
# OUTPUT: Q Data:
# tensor([[ 2.0803, -1.1089,  2.6151,  0.6214],
#         [-0.3051, -0.0085,  2.7119, -1.4031],
#         [ 0.4176, -0.0462,  1.9627,  0.6518]])

print(f"K Shape: {list(K.size())}")
print(f"K Data:\n{K[0]}")
# OUTPUT: K Shape: [1, 3, 4]
# OUTPUT: K Data:
# tensor([[-1.8736, -0.4785,  3.8062, -0.5980],
#         [-3.3101,  1.3053, -0.4011,  5.3528],
#         [-0.0603, -1.5544, -1.1237, -1.0270]])

print(f"V Shape: {list(V.size())}")
print(f"V Data:\n{V[0]}")
# OUTPUT: V Shape: [1, 3, 4]
# OUTPUT: V Data:
# tensor([[-1.7691, -1.5515,  0.5958,  0.8932],
#         [-1.9528,  1.6973, -3.5038,  1.2586],
#         [ 1.4267,  0.9428, -3.0853, -0.5809]])


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
print(f"Head 1 Q Data:\n{Q_h1[0]}")
# OUTPUT: Head 1 Q Shape: [1, 3, 2]
# OUTPUT: Head 1 Q Data:
# tensor([[ 2.0803, -1.1089],
#         [-0.3051, -0.0085],
#         [ 0.4176, -0.0462]])

print(f"Head 2 Q Shape: {list(Q_h2.size())}")
print(f"Head 2 Q Data:\n{Q_h2[0]}")
# OUTPUT: Head 2 Q Shape: [1, 3, 2]
# OUTPUT: Head 2 Q Data:
# tensor([[ 2.6151,  0.6214],
#         [ 2.7119, -1.4031],
#         [ 1.9627,  0.6518]])


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
# OUTPUT: Head 1 Attention Probabilities (3x3):
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.4258, 0.5742, 0.0000],
#         [0.2953, 0.1822, 0.5225]])

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
print(f"Concatenated Output Data:\n{concat_output[0]}")
# OUTPUT: Concatenated Output Shape: [1, 3, 4]
# OUTPUT: Concatenated Output Data:
# tensor([[-1.7691, -1.5515,  0.5958,  0.8932],
#         [-1.8746,  0.3141,  0.5958,  0.8932],
#         [-0.1328,  0.3438,  0.4156,  0.9078]])

# Final linear projection (mixing the heads context via matrix multiplication)
W_proj_attn = torch.randn(C, C)
attn_final = concat_output @ W_proj_attn
print(f"attn_final Shape: {list(attn_final.size())}")
print(f"attn_final Data:\n{attn_final[0]}")
# OUTPUT: attn_final Shape: [1, 3, 4]
# OUTPUT: attn_final Data:
# tensor([[ 0.6096,  2.5242, -1.9060, -0.8313],
#         [ 0.9619,  0.3654, -1.2757,  2.3537],
#         [ 0.7054,  0.6844, -1.2558,  2.0945]])

# ==============================================================================
# 6. RESIDUAL CONNECTION 1
# ==============================================================================
print("\n--- 6. RESIDUAL CONNECTION 1 ---")
x = x + attn_final # The original word + its new context
print(f"x Shape after Residual 1: {list(x.size())}")
print(f"x Data after Residual 1:\n{x[0]}")
# OUTPUT: x Shape after Residual 1: [1, 3, 4]
# OUTPUT: x Data after Residual 1:
# tensor([[ 0.9463,  2.6530, -1.6715, -0.6009],
#         [-0.1609,  0.1790,  0.9325,  1.7157],
#         [ 1.1671,  0.9517, -0.7209,  2.9039]])


# ==============================================================================
# 7. LAYER NORMALIZATION 2 (Manual)
# ==============================================================================
print("\n--- 7. LAYER NORMALIZATION 2 ---")
mean = x.mean(dim=2, keepdim=True)
variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)
norm_x = (x - mean) / torch.sqrt(variance + epsilon)

print(f"norm_x Shape after LayerNorm 2: {list(norm_x.size())}")
print(f"norm_x Data after LayerNorm 2:\n{norm_x[0]}")
# OUTPUT: norm_x Shape after LayerNorm 2: [1, 3, 4]
# OUTPUT: norm_x Data after LayerNorm 2:
# tensor([[ 0.3767,  1.4227, -1.2277, -0.5716],
#         [-1.1437, -0.6739,  0.3675,  1.4501],
#         [ 0.0714, -0.0964, -1.3991,  1.4241]])

# ==============================================================================
# 8-10. MLP (Manual Math, No nn.Linear)
# ==============================================================================
print("\n--- 8. MULTI-LAYER PERCEPTRON (Manual Math) ---")
print("Focusing on Word 3 (The blended context of 'Cat is cute')")

# A. Expansion (Multiply by a 4x16 weight matrix)
W_expand = torch.randn(C, 4 * C) # Expand from 4 to 16 dimensions
mlp_hidden = norm_x @ W_expand
print(f"Expanded Shape: {list(mlp_hidden.size())}")
print(f"Expanded Data (mlp_hidden[0]):\n{mlp_hidden[0]}")
# OUTPUT: Expanded Shape: [1, 3, 16]
# OUTPUT: Expanded Data (mlp_hidden[0]):
# tensor([[-1.7510,  1.4493, -1.2454, -0.2088, -1.7796, -0.3460,  0.0715,  1.0359,
#          -0.8983,  0.0575,  2.3617,  1.8431,  0.3265,  0.3632,  5.8125,  1.1895],
#         [-0.5198,  0.0789,  1.0407,  0.2080,  2.0873,  1.7950, -1.1089, -1.0499,
#           0.9954, -1.4150, -3.8893, -1.5305, -2.3959,  1.5637, -4.8072, -0.5763],
#         [-1.8392,  0.7937, -1.3743,  1.7317,  1.4434, -0.2047, -1.1668,  0.0653,
#           1.1929, -0.6969, -1.3229, -0.1709, -0.8432,  1.0423, -3.3733,  0.0423]])

print(f"\nExpanded 16D 'Scratchpad' for Word 3 (Before GELU):\n{mlp_hidden[0, 2]}")
# OUTPUT: Expanded 16D 'Scratchpad' for Word 3 (Before GELU):
# tensor([-1.8392,  0.7937, -1.3743,  1.7317,  1.4434, -0.2047, -1.1668,  0.0653,
#          1.1929, -0.6969, -1.3229, -0.1709, -0.8432,  1.0423, -3.3733,  0.0423])

# B. GELU Activation
# We use standard PyTorch GELU here as the manual polynomial approximation is very long
mlp_activated = torch.nn.functional.gelu(mlp_hidden)

print(f"\nActivated 16D 'Scratchpad' for Word 3 (After GELU):\n{mlp_activated[0, 2]}")
# OUTPUT: Activated 16D 'Scratchpad' for Word 3 (After GELU):
# tensor([-6.0590e-02,  6.2407e-01, -1.1637e-01,  1.6596e+00,  1.3360e+00,
#         -8.5764e-02, -1.4194e-01,  3.4369e-02,  1.0540e+00, -1.6930e-01,
#         -1.2294e-01, -7.3850e-02, -1.6827e-01,  8.8734e-01, -1.2530e-03,
#          2.1856e-02])
print("Notice how GELU crushed negative values near zero, and kept positive ones!")

# C. Compression (Multiply by a 16x4 weight matrix)
W_compress = torch.randn(4 * C, C) # Compress back to 4 dimensions
mlp_final = mlp_activated @ W_compress
print(f"\nCompressed Shape: {list(mlp_final.size())}")
print(f"Compressed Data (mlp_final[0]):\n{mlp_final[0]}")
# OUTPUT: Compressed Shape: [1, 3, 4]
# OUTPUT: Compressed Data (mlp_final[0]):
# tensor([[ -2.9948,  -7.9997, -11.8647,  -6.9604],
#         [  1.3216,   1.1039,  -2.0342,   1.5107],
#         [  0.2672,   2.1692,   1.3576,   0.9292]])

print(f"Final 4D Compressed Meaning for Word 3:\n{mlp_final[0, 2]}")
# OUTPUT: Final 4D Compressed Meaning for Word 3:
# tensor([0.2672, 2.1692, 1.3576, 0.9292])

# ==============================================================================
# 11. RESIDUAL CONNECTION 2
# ==============================================================================
print("\n--- 11. RESIDUAL CONNECTION 2 ---")
x = x + mlp_final

print(f"FINAL OUTPUT SHAPE OF THE BLOCK: {list(x.size())}")
# OUTPUT: FINAL OUTPUT SHAPE OF THE BLOCK: [1, 3, 4]

print(f"Final Output Data x[0]:\n{x[0]}")
# OUTPUT: Final Output Data x[0]:
# tensor([[ -2.0485,  -5.3466, -13.5362,  -7.5613],
#         [  1.1607,   1.2829,  -1.1017,   3.2264],
#         [  1.4343,   3.1209,   0.6367,   3.8331]])
