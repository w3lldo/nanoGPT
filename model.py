"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.
    Layer Normalization normalizes the output of a layer (mean 0, variance 1)
    and scales/shifts it using learned parameters. This custom class adds a toggle
    for the bias parameter for optimization purposes.
    """

    def __init__(self, ndim, bias):
        """
        Initializes the LayerNorm with learnable Gamma (weight) and Beta (bias) parameters.
        """
        super().__init__()
        # self.weight: The "Gamma" scale parameter. Initializes to all 1s so it doesn't change anything at first. It will be learned to stretch/squish data.
        self.weight = nn.Parameter(torch.ones(ndim))
        # self.bias: The "Beta" shift parameter. Initializes to 0s so it doesn't change anything at first. It will be learned to shift data left/right.
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        Executes the LayerNorm mathematical operation: (x - mean) / variance * scale + shift.

        GEOMETRIC/TYPE EXPLANATION:
        `input` is a `torch.Tensor` typically of shape (B, T, C). It represents the batch of embedded sentences.
        """
        # F.layer_norm: The actual math. Forces the mean of the input to 0 and variance to 1, then applies our learned weight (scale) and bias (shift). 1e-5 prevents dividing by zero.
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    The heart of the Transformer. Calculates multi-head masked self-attention.
    Projects the input into Queries, Keys, and Values. It applies a causal mask
    so that tokens can only attend to previous tokens, not future ones, ensuring
    autoregressive generation. Takes a weighted sum of the Values based on Attention scores.
    """

    def __init__(self, config):
        """
        Sets up the Linear projections for Queries, Keys, and Values, and initializes Dropouts and the Causal Mask.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # self.c_attn: A single massive Linear layer that creates the Queries, Keys, and Values all at once! This is 3x the embedding size so it can be split cleanly into 3 equal pieces later.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # self.c_proj: The final Linear layer that mixes the output of all the parallel attention heads back together into a single 768-D vector.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Executes the Multi-Head Attention mechanism.
        Splits data into Q, K, V matrices, applies the causal mask to prevent cheating, calculates
        attention probabilities via Softmax, and weights the Values to produce the final context updates.

        GEOMETRIC/TYPE EXPLANATION:
        `x` is a `torch.Tensor` of shape `(B, T, C)` exactly like in the main Block.
        (Batch Size, Sequence Length, and Embedding Dimensions).
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # c_attn(x): Projects the input 'x' into the massive QKV matrix. .split(): slices that massive matrix into 3 exact 1/3rd chunks: Queries, Keys, and Values.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # .view(): Splits the 768-dimension features perfectly into 12 separate "heads" of 64 features each. .transpose(): Rearranges dimensions so the Head is the 2nd dimension for efficient parallel processing.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # (q @ k.transpose): Words asking questions (Q) multiplied by words answering them (K) to find "Similarity Scores" (how much they should care about each other).
            # * (1.0 / math.sqrt...): Scales the scores down so they don't explode and break the softmax.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # masked_fill: The "Causal" part! Forces all future word combinations to negative infinity (-inf) so words can NEVER cheat by looking at answers they haven't seen yet.
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # F.softmax: Converts the raw similarity scores into percentages (probabilities that sum to 1.0).
            att = F.softmax(att, dim=-1)
            # attn_dropout: Randomly zeroes out some attention scores during training to prevent memorization / overfitting.
            att = self.attn_dropout(att)
            # att @ v: The final weighting! We multiply our newly calculated percentages against the actual Values (V) to get the final context-mixed updates.
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # transpose.contiguous.view: Re-assembles the partitioned heads back together. The 12 heads of 64 features are slammed back into a single continuous 768-D vector.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # c_proj(y): The final Linear matrix that mixes the newly assembled vector.
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    A standard Feed-Forward Neural Network (Multi-Layer Perceptron).
    Acts as the 'thinking' part of the transformer block. After attention mixes
    information across the sequence, the MLP processes that combined information
    individually per token. Uses a GELU activation function.
    """

    def __init__(self, config):
        """
        Initializes the fully-connected expansion layer, the GELU activation, and the compression layer.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Processes information for an individual token. Expands the 768-D representation to 3072-D for
        deep processing, applies non-linear GELU activation, and compresses it back down to 768-D.

        GEOMETRIC/TYPE EXPLANATION:
        `x` is a `torch.Tensor` of shape `(B, T, C)`. Although the MLP mathematically processes each Token
        completely independently (it doesn't look at neighbor words like Attention does), the Batch (B) 
        and Time (T) dimensions are still carried straight through the math.
        """
        # c_fc(x): The "Expansion" phase. Projects the 768-D input up to a massive 3072-D vector so it can "think" about concepts in high dimensionality.
        x = self.c_fc(x)
        # gelu(x): The Non-Linear activation. Without this, the network would just be doing flat linear algebra. This allows it to learn complex, curvy patterns.
        x = self.gelu(x)
        # c_proj(x): The "Compression" phase. Squeezes the expanded 3072-D thought back down perfectly to the 768-D vector shape so it can fit smoothly back onto the Super-Highway.
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    The fundamental repeating unit of a Transformer. A GPT model is primarily a stack
    of these blocks. Each block applies Layer Normalization, multi-head Causal Self-Attention
    to mix information across the sequence, and an MLP to process the mixed information,
    along with residual connections.
    """

    def __init__(self, config):
        """
        Initializes the fundamental Transformer components: Attention, MLP, and their respective LayerNorms.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Runs the full block logic: Takes data from the residual highway, passes it through Attention
        for context, adds it back to the highway. Then passes the new data through the MLP for "thinking",
        and adds it back to the highway again.

        WHAT IS `x` GEOMETRICALLY?
        Because Python's dynamic typing hides it, it's important to know `x` is a `torch.Tensor` (a highly 
        optimized 3D array of floating point numbers). Its shape is exactly (B, T, C), meaning:
        - B (Batch): Number of independent sentences processed at once (e.g., 12).
        - T (Time): Sequence length / number of words in the sentence (e.g., 1024).
        - C (Channels): Embedding size / math features defining a single word (e.g., 768).
        If you isolated one word `x[0, 5, :]`, it is literally an array of 768 decimal numbers that represent its current meaning!
        """
        # THE CALCULUS OF `+` (THE RESIDUAL SUPER-HIGHWAY):
        # In the forward pass, this adds the new context mathematically back into the original 'x' identity.
        # In the BACKWARD pass (learning), the `+` acts as a perfect "Gradient Distributor". 
        # When an error signal hits the `+`, it gets perfectly cloned. One copy goes into the Attention/MLP 
        # to update those specific weights. The other copy rockets STRAIGHT DOWN the pristine 'x' highway 
        # completely untouched, bypassing the complex math! This prevents the "Vanishing Gradient Problem" 
        # and guarantees that early layers always receive a loud, clear learning signal.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """
    A blueprint/settings class holding hyperparameters for the GPT model.
    Defines the depth (n_layer), width (n_embd), attention heads (n_head),
    vocab size, and maximum sequence length (block_size).
    """
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    """
    The main GPT model container. It ties everything together:
    - Word Token Embeddings (wte) and Word Position Embeddings (wpe)
    - A stack of Transformer Blocks
    - A final Output Head (lm_head) to project back to the vocabulary size
    It defines how data flows from input tokens to predicted logits.
    """

    def __init__(self, config):
        """
        Initializes the ENTIRE GPT model, including vocabulary embeddings, positional embeddings,
        the stack of transformer blocks, and the final language modeling head.
        """
        # super().__init__(): Initializes the parent PyTorch nn.Module class.
        super().__init__()
        # assert...: Checks that we actually provided a vocabulary size and a maximum sequence length (block size).
        assert config.vocab_size is not None
        assert config.block_size is not None
        # self.config = config: Saves the configuration settings inside the object for later use.
        self.config = config

        # self.transformer: Creates a dictionary of PyTorch layers that holds the core pieces of the model.
        self.transformer = nn.ModuleDict(dict(
            # wte: Word Token Embeddings. Converts a word index into a dense feature vector.
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe: Word Position Embeddings. Gives a vector representing positions, so the model knows word order.
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop: Dropout layer applied after adding word and position embeddings to prevent overfitting.
            drop = nn.Dropout(config.dropout),
            # h: The "hidden" layers. A stack of identical Transformer Blocks.
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f: The final Layer Normalization applied after the data exits the last Block.
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # lm_head: The Language Model Head. Projects the final feature vector back out to the vocabulary size to get probabilities.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight Tying: The word embeddings and the final output head share the exact same weights matrix, saving 38M parameters!
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # sum(p.numel()...): Adds up the total count of numbers/variables the model is trying to learn.
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtracts the positional embeddings from the count, following standard conventions for reporting model sizes.
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initialization logic for setting the initial untrained values (Gaussian random distributions)
        for all Linear and Embedding modules before training begins.
        """
        if isinstance(module, nn.Linear):
            # If the layer is a Linear mathematical projection, initialize the weights to a Normal/Gaussian distribution.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # If the layer has bias parameters, initialize them to exactly 0.
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # If the layer is an Embedding dictionary, initialize the weights identically to the Linear modules.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        The Main Forward Pass. Takes raw word tokens, embeds them, creates positional vectors,
        runs them through the massive Transformer Block stack, and finally maps the dense 768-D vectors
        back into 50,000+ vocabulary probabilities using the LM Head. Calculates Loss during training.

        GEOMETRIC/TYPE EXPLANATION:
        `idx` and `targets` are `torch.Tensor` objects of shape `(B, T)` (Batch, Time).
        CRITICALLY: Unlike internal blocks, these do NOT contain floating point numbers! They contain 
        raw Integer IDs representing words (e.g., `idx[0,:] = [12, 532, 1024...]`).
        This function returns `logits` formatted as a massively dense float Tensor of shape `(B, T, Vocab_Size)`.
        """
        device = idx.device
        # Extracts the Batch (b) and Sequence length (t) from the input.
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # torch.arange(): Creates a simple array of numbers from 0 to t to represent the position indices.
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # wte(idx): Converts raw token IDs into dense feature vectors.
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # wpe(pos): Converts raw positions into feature vectors.
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # Adds them together! 'x' now mathematically represents "What is this word, and where is it?", and applies dropout.
        x = self.transformer.drop(tok_emb + pos_emb)
        # The Super-Highway! The data 'x' flows sequentially through all blocks.
        for block in self.transformer.h:
            x = block(x)
        # Applies the final LayerNorm after emerging from the deep stack of blocks.
        x = self.transformer.ln_f(x)

        if targets is not None:
            # During Training (targets given): Pass the full highway data into the final lm_head to get the logits (predictions).
            logits = self.lm_head(x)
            # Calculate the loss by comparing those predictions against the actual targets.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # During Generation (inference-time optimization): Pluck out only the very last position in the sequence and pass that through the head to save computation.
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        Model surgery logic. Useful if we load a massive pre-trained model (like GPT-2 1024 block size)
        but only have enough GPU memory to run a 256 block size context window. Slices the arrays.
        """
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Slices the Position Embedding dictionary to throw out absolute positions we don't need anymore.
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # Shrinks the geometric Causal Mask (the triangle) in every attention block so it tightly matches the new smaller sequence length.
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        The "State Dictionary Hijacker"! 
        This method doesn't train a model. Instead, it creates an empty 'shell' of our GPT model, 
        downloads full, pre-trained weights from HuggingFace (originally trained by OpenAI for millions of dollars),
        and surgically injects those weights into our empty shell so we can chat with it immediately!
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 1. ARCHITECTURE MAPPING: Determine the exact shape of the "empty shell" we need to build based on the model name.
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        # Hardcode the static assumptions built into OpenAI's original GPT-2 release:
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        # 2. CREATE OUR EMPTY SHELL
        config = GPTConfig(**config_args)
        model = GPT(config)
        # Grab our empty dictionary of random weights that we just initialized
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Discard the Causal Mask buffer from the list of weights to copy, because it's just a generated triangle of 1s and 0s, not a learned training parameter.
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 3. DOWNLOAD THE HUGGINGFACE WEIGHTS
        # This reaches out to the internet, downloads the ~500MB+ PyTorch checkpoint file, and loads it into memory.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 4. PREPARE THE SURGERY
        # Grab HuggingFace's dictionary of fully-trained weights
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # OpenAI used a weird "Conv1D" module instead of standard "Linear" math for their matrices when they published the code back in 2019.
        # Conv1D stores matrices transposed (flipped sideways) compared to how our clean `nn.Linear` expects them.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Ensure our list of empty dictionary layers perfectly lines up with HuggingFace's list of completed layers.
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        # 5. EXECUTE THE SURGERY
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # If we hit one of OpenAI's weird Conv1D matrices, we must verify the flip shape
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    # .t() transposes (flips) the HuggingFace matrix geometry so it perfectly fills our nn.Linear matrix geometry! 
                    sd[k].copy_(sd_hf[k].t())
            else:
                # For standard weights (like LayerNorms or Embeddings), the geometries already match perfectly.
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    # Simply copy the trained numbers over into our empty shell.
                    sd[k].copy_(sd_hf[k])

        # Return the freshly loaded, fully trained model!
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Sets up the AdamW optimization strategy. 
        Intelligently separates massive weight matrices (which require weight decay regularization)
        from 1D biases/LayerNorms (which do not), then initializes the optimizer to begin training.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # Decay params: Massive matrices are given "weight decay" to prevent them from growing too large and memorizing the data.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # No-decay params: 1D vectors (like Biases or LayerNorm scales) are NOT punished for getting large.
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        GEOMETRIC/TYPE EXPLANATION:
        `idx` starts as a `torch.Tensor` of integers of shape `(B, T)`. 
        With each loop iteration, we predict 1 new token mathematically, and literally glue it 
        onto the end of the array using `torch.cat`, causing the sequence length (T) to grow by 1 each time!
        Returns a final integer tensor of shape `(B, T + max_new_tokens)`.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size, since the network physically cannot look back further than block_size!
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature (High temp = more chaotic choices, Low temp = rigid choices)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to percentages/probabilities
            probs = F.softmax(logits, dim=-1)
            # sample randomly from the distribution (e.g. 80% chance it picks "dog", 20% it picks "cat")
            idx_next = torch.multinomial(probs, num_samples=1)
            # glue that newly chosen word onto the end of your original sentence array, and repeat the loop!
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
