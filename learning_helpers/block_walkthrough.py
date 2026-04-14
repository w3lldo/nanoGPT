import torch
import torch.nn.functional as F
import math

# ==============================================================================
# 0. SETUP THE DATA
# ==============================================================================
print("\n--- 0. SETUP THE DATA ---")

# Let's say we have a sentence with 3 words (Sequence Length T=3)
# Each word is represented by a 4-dimensional vector (Embedding Size C=4)
# Batch size B=1
B, T, C = 1, 3, 4

# This is our input "x". 
# For example, word 1 might be [0.1,  0.5, -0.2,  0.9].
# In reality, this comes from Word Embeddings + Position Embeddings.
x = torch.randn(B, T, C)

print(f"Input x Shape: {list(x.size())}")
print(f"Input x Data:\n{x[0]}")

# ==============================================================================
# PHASE 1: THE COMMUNICATION PHASE 
# (Words talk to each other to understand context)
# ==============================================================================
print("\n--- PHASE 1: COMMUNICATION (Self-Attention) ---")

# 1. Layer Normalization (ln_1)
# PyTorch has this built-in. It normalizes each word vector (across the 4 dims) 
# so they have mean 0 and variance 1, preventing any single feature from dominating.
ln_1_weight = torch.ones(C)
ln_1_bias = torch.zeros(C)
norm_x = F.layer_norm(x, (C,), ln_1_weight, ln_1_bias, 1e-5)

print("\n1. After LayerNorm 1 (norm_x):")
print(f"Shape: {list(norm_x.size())}")
print("Notice how the values are now stabilized around 0.")
print(f"Data:\n{norm_x[0]}")

# 2. Multi-Head Self Attention (c_attn)
# We need Queries, Keys, and Values. We project our 4D input into a 12D tensor.
# (3 concatenated 4D tensors for Q, K, and V).
c_attn_weight = torch.randn(C, 3 * C) 
qkv = norm_x @ c_attn_weight

# Now we split that 12D tensor into our separate Q, K, and V tensors.
q, k, v = qkv.split(C, dim=2)

print("\n2. Projecting to Q, K, V:")
print(f"Q Shape: {list(q.size())}")

# 3. Splitting into Heads
# We use 2 Attention Heads. We chop our 4 features into 2 heads, 
# so each head gets 2 features (Head Size = 2).
n_head = 2
head_size = C // n_head # 4 // 2 = 2

# Reshape from (B, T, C) -> (B, n_head, T, head_size)
q = q.view(B, T, n_head, head_size).transpose(1, 2)
k = k.view(B, T, n_head, head_size).transpose(1, 2)
v = v.view(B, T, n_head, head_size).transpose(1, 2)

print("\n3. Splitting into 2 Heads:")
print(f"Q Shape after split: {list(q.size())} -> (Batch, Heads, Time, Head_Size)")

# 4. Attention Calculation (The part you wrote your article about!)
# Calculate dot product: Q @ K^T
att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))

# Apply Causal Mask (prevent looking at future words)
mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
att_scores = att_scores.masked_fill(mask == 0, float('-inf'))

# Softmax to get probabilities
att_probs = F.softmax(att_scores, dim=-1)

# Weighted sum of Values
att_output = att_probs @ v  # Shape is now (B, n_head, T, head_size): (1, 2, 3, 2)

print("\n4. After Attention Calculation:")
print(f"Attention Output Shape: {list(att_output.size())}")

# 5. Re-assemble and Project (c_proj)
# We glue the 2 heads back together to get back to our 4D space.
# Shape goes from (1, 2, 3, 2) back to (1, 3, 4)
att_output = att_output.transpose(1, 2).contiguous().view(B, T, C)

# A final linear projection to mix the heads
c_proj_weight = torch.randn(C, C)
attn_final = att_output @ c_proj_weight

print("\n5. After Re-assembling Heads and Final Projection:")
print(f"attn_final Shape: {list(attn_final.size())}")

# 6. Residual Connection 1
# We ADD the original input 'x' to the new 'attn_final' context.
# This ensures we don't forget the original word meaning!
x = x + attn_final

print("\n6. After Residual Connection 1 (x = x + attn):")
print(f"Shape: {list(x.size())}")
print("This completes the Communication Phase!")

# ==============================================================================
# PHASE 2: THE THINKING PHASE 
# (Words reflect individually on the new context)
# ==============================================================================
print("\n--- PHASE 2: THINKING (MLP) ---")

# 7. Layer Normalization (ln_2)
# Stabilize again before the heavy lifting.
ln_2_weight = torch.ones(C)
ln_2_bias = torch.zeros(C)
norm_x = F.layer_norm(x, (C,), ln_2_weight, ln_2_bias, 1e-5)

print("\n7. After LayerNorm 2 (norm_x):")
print(f"Shape: {list(norm_x.size())}")

# 8. MLP Expansion (c_fc)
# We expand the 4D feature vector by 4x. This acts like a "working memory" 
# scratchpad for the network to combine patterns.
# Shape expands from (1, 3, 4) -> (1, 3, 16)
mlp_expand_weight = torch.randn(C, 4 * C)
mlp_expanded = norm_x @ mlp_expand_weight

print("\n8. MLP Expansion (c_fc):")
print(f"Expanded Shape: {list(mlp_expanded.size())}")

# 9. Non-Linear Activation (GELU)
# This allows the network to learn complex, non-linear relationships.
mlp_activated = F.gelu(mlp_expanded)

# 10. MLP Compression (c_proj)
# We compress the 16D scratchpad back down into the standard 4D space.
# Shape compresses from (1, 3, 16) -> (1, 3, 4)
mlp_compress_weight = torch.randn(4 * C, C)
mlp_final = mlp_activated @ mlp_compress_weight

print("\n10. MLP Compression (c_proj):")
print(f"Compressed final shape: {list(mlp_final.size())}")

# 11. Residual Connection 2
# We ADD the original input to the new MLP deep thoughts.
x = x + mlp_final

print("\n11. After Residual Connection 2 (x = x + mlp):")
print(f"FINAL OUTPUT SHAPE OF THE BLOCK: {list(x.size())}")
print(f"Final Data (The fully context-aware word vectors):\n{x[0]}")

print("\nThis tensor is now ready to be fed into Transformer Block 2!")
