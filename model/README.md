# 🧠 model/ — Modular GPT Implementation

This directory contains the modular components of a simplified GPT (Generative Pretrained Transformer) language model. Inspired by [Andrej Karpathy's](https://github.com/karpathy/ng-video-lecture) tutorial, this version breaks down the model into individual Python files to improve readability and understanding.

## 📁 Modules Overview

### `config.py`
This file stores all the global hyperparameters used throughout the model and training process. These include:

- `batch_size`: How many samples are processed in parallel.
- `block_size`: How many tokens the model looks at for context.
- `n_embd`: Embedding dimension for each token (e.g., 384).
- `n_head`: Number of attention heads.
- `n_layer`: Number of Transformer blocks.
- `dropout`: Dropout probability.
- `learning_rate`, `max_iters`, etc.

> 🧠 *My understanding*: "Batch size is how many samples are sent in each time, and each example is a 256×384 vector — 256 tokens, each with 384 features."

---

### `gpt_model.py`
This is the main class `GPTLanguageModel`, which ties everything together.

- Initializes **token embeddings** and **position embeddings**
- Stacks multiple `Block` modules (attention + feedforward)
- Applies a final `LayerNorm` and a linear layer (`lm_head`) to produce logits over the vocabulary
- Handles the forward pass with optional loss computation
- Includes a `.generate()` method for autoregressive text generation

> 🧠 *My observation*: "This class just combines everything: token embedding, position embedding, attention blocks."

---

### `block.py`
Defines the `Block` class — a standard Transformer block.

Each block contains:
- A `MultiHeadAttention` module
- A `FeedForward` module
- Two `LayerNorm` layers
- Residual connections for stability

> 🧠 *My understanding*: "Block is attention + feedforward, with layer norms applied and residuals used to avoid vanishing gradients."

---

### `attention.py`
Defines:
- `Head`: A single self-attention head, implementing queries, keys, values, causal masking with a lower-triangular matrix (`register_buffer('tril', ...)`)
- `MultiHeadAttention`: Combines multiple `Head` instances in parallel and projects the result back to the embedding size.

> 🧠 *My summary*: "This line `self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])` creates 6 attention heads. After that, everything gets concatenated and projected back to (64, 256, 384) — not by reducing size, but by transforming the last dimension linearly."

---

### `feedforward.py`
Defines the `FeedForward` class — a standard 2-layer MLP with ReLU and dropout.

- Input and output dimensions match the embedding size (`n_embd`)
- Intermediate hidden layer is typically 4× wider (`4 * n_embd`)

---

## 🛠 Integration Notes
- All modules rely on `config.py` for shared hyperparameters
- You can swap components or change architecture depth by editing `config.py` and `gpt_model.py`

## ✅ Why Modular?
Breaking the code into small components:
- Makes it easier to understand and debug
- Helps you learn each part of the transformer individually
- Encourages better code reuse and flexibility

---

Feel free to explore and modify each part to deepen your understanding of transformer architectures!
