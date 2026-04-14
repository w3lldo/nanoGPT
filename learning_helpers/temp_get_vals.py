import torch
import math

torch.manual_seed(42)
B, T, C = 1, 3, 4
x = torch.randn(B, T, C)
epsilon = 1e-5
mean = x.mean(dim=2, keepdim=True) 
variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)
norm_x = (x - mean) / torch.sqrt(variance + epsilon)

W_q = torch.randn(C, C) 
W_k = torch.randn(C, C) 
W_v = torch.randn(C, C) 

Q = norm_x @ W_q
K = norm_x @ W_k
V = norm_x @ W_v

print('# OUTPUT: Q Data:\n# ' + str(Q[0]).replace('\n', '\n# '))
print('# OUTPUT: K Data:\n# ' + str(K[0]).replace('\n', '\n# '))
print('# OUTPUT: V Shape: ' + str(list(V.size())))
print('# OUTPUT: V Data:\n# ' + str(V[0]).replace('\n', '\n# '))

Q_h1 = Q[:, :, 0:2] 
K_h1 = K[:, :, 0:2]
V_h1 = V[:, :, 0:2]
Q_h2 = Q[:, :, 2:4] 
K_h2 = K[:, :, 2:4]
V_h2 = V[:, :, 2:4]

print('# OUTPUT: Head 1 Q Data:\n# ' + str(Q_h1[0]).replace('\n', '\n# '))
print('# OUTPUT: Head 2 Q Data:\n# ' + str(Q_h2[0]).replace('\n', '\n# '))

d_k = 2
scores_h1 = Q_h1 @ K_h1.transpose(1, 2) / math.sqrt(d_k)
scores_h1[0, 0, 1] = float('-inf') 
scores_h1[0, 0, 2] = float('-inf') 
scores_h1[0, 1, 2] = float('-inf') 

probs_h1 = torch.exp(scores_h1) / torch.exp(scores_h1).sum(dim=-1, keepdim=True)
output_h1 = probs_h1 @ V_h1

scores_h2 = (Q_h2 @ K_h2.transpose(1, 2)) / math.sqrt(d_k)
scores_h2[0, 0, 1] = float('-inf')
scores_h2[0, 0, 2] = float('-inf')
scores_h2[0, 1, 2] = float('-inf')
probs_h2 = torch.exp(scores_h2) / torch.exp(scores_h2).sum(dim=-1, keepdim=True)
output_h2 = probs_h2 @ V_h2

concat_output = torch.cat((output_h1, output_h2), dim=2)
print('# OUTPUT: Concatenated Output Data:\n# ' + str(concat_output[0]).replace('\n', '\n# '))

W_proj_attn = torch.randn(C, C)
attn_final = concat_output @ W_proj_attn
print('# OUTPUT: attn_final Shape: ' + str(list(attn_final.size())))
print('# OUTPUT: attn_final Data:\n# ' + str(attn_final[0]).replace('\n', '\n# '))

x = x + attn_final
print('# OUTPUT: x Data after Residual 1:\n# ' + str(x[0]).replace('\n', '\n# '))

mean = x.mean(dim=2, keepdim=True)
variance = ((x - mean) ** 2).mean(dim=2, keepdim=True)
norm_x = (x - mean) / torch.sqrt(variance + epsilon)

print('# OUTPUT: norm_x Shape after LayerNorm 2: ' + str(list(norm_x.size())))
print('# OUTPUT: norm_x Data after LayerNorm 2:\n# ' + str(norm_x[0]).replace('\n', '\n# '))

W_expand = torch.randn(C, 4 * C) 
mlp_hidden = norm_x @ W_expand
print('# OUTPUT: Expanded Data (mlp_hidden[0]):\n# ' + str(mlp_hidden[0]).replace('\n', '\n# '))

mlp_activated = torch.nn.functional.gelu(mlp_hidden)
W_compress = torch.randn(4 * C, C) 
mlp_final = mlp_activated @ W_compress
print('# OUTPUT: Compressed Data (mlp_final[0]):\n# ' +  str(mlp_final[0]).replace('\n', '\n# '))
