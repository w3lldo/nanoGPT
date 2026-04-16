# Project Roadmap: TabPFN Foundation Model & Mechanistic Interpretability
**Time Commitment:** ~2-3 hours nightly (~15-20 hours/week)
**Overall Timeline:** 4 Weeks
**Goal:** Build, train, and mechanically probe a dummy Tabular Foundation Model (TabPFN) from scratch on Apple Silicon, entirely bypassing language architecture constraints.

---

## Week 1: The Architecture Surgery (Model Engineering)
*Context: Stripping away language biases from the Transformer and building raw numerical ingestion engines.*

*   **Day 1-2:** Cleanly duplicate the core `Block` structures from `nanoGPT` into a new `tab_model.py` file to preserve the learning sandbox.
*   **Day 3-4:** Amputate the `wte` (Word Token) and `wpe` (Positional) embedding dictionaries. Replace them with standard dense `nn.Linear`/MLP un-embedding projectors to accept raw arrays of floats.
*   **Day 5-6:** Redesign the `CausalSelfAttention` masking logic. TabPFN requires targets to see contexts (In-Context Learning), but context rows shouldn't cheat by looking at targets.
*   **Day 7 (Deliverable):** A fully compiling `tab_model.py` script that smoothly runs dummy, mathematically-random tabular tensors through the Forward and Backward passes without shape mismatch errors.

---

## Week 2: The Data Pipeline & Training Hooks
*Context: Connecting the brain to the food supply. We must rewrite how data is organized and batched.*

*   **Day 8-10:** Secure and integrate an open-source Tabular Prior mathematical generator (Bayesian Networks/SCMs) to avoid spending 2 months writing equations from scratch.
*   **Day 11-12:** Completely rewrite the `get_batch()` function in `train.py`. The batch must now construct distinct "Tasks" consisting of `[X_context, Y_context, X_target, Y_target]` rather than sequential text.
*   **Day 13-14:** Wire the newly generated Tabular batches directly into the modified `train.py` loop. Manage `bfloat16` and Gradient clipping to ensure extreme math volatility doesn't cause `NaN` crashes.
*   **Week 2 Deliverable:** The model actively trains for multiple hours on the Mac `mps` GPU, with the Loss Graph slowly moving downwards. 

---

## Week 3: Mac GPU Execution & Tensor Caching
*Context: Letting the Mac do the heavy lifting and setting up the surgical extraction suite.*

*   **Day 15-18 (Passive):** Launch the primary `train.py` script and leave the Mac running on full GPU output to train the foundational priors. Intervene only to adjust learning rates.
*   **Day 19-20:** Once the model finishes training, freeze the neural network using `model.eval()` and `torch.no_grad()`. 
*   **Day 21:** Build PyTorch "Hooks" (`register_forward_hook`). These intercept the data `x` as it physically travels across the Residual Super-Highway inside the frozen model. Dump these extracted mathematical "thoughts" into raw `.pt` files on your hard drive. 
*   **Week 3 Deliverable:** A completely frozen, highly intelligent mini-TabPFN, and a hard drive full of its cached internal activation states.

---

## Week 4: Mechanistic Probing (Anthropic SAEs)
*Context: The pure research phase. Dissecting the frozen model to see *how* it thinks.*

*   **Day 22-23:** Build the Linear Probe pipeline. Feed the extracted layer caches into a single standard Logistic Regression layer. Run queries like: *"Does Layer 6 inherently track if a feature is normally distributed?"*
*   **Day 24-26:** Code the Sparse Autoencoder (SAE). Extract the internal `512-D` embedding spaces, expand them to `4096-D` with `L1` sparsity penalties, and shrink them back down to decode the "Superposition" of neurons multi-tasking tabular concepts.
*   **Day 27-28:** Analyze and visualize the probing results/SAE feature directions. Compare lower block layers against deeper block layers to track how abstract reasoning evolved.
*   **Day 30 (Deliverable):** Final evaluation of the Probing Accuracy, confirming the feasibility of deploying this architecture as a Sports Pricing Engine next!
