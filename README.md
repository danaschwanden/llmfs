# LLM from Scratch : Study & Implementation Plan

This document serves as a structured notepad and roadmap for unpacking, implementing, and mastering the architectural and mathematical concepts of the [GPT-2](https://openai.com/index/better-language-models/) large [transformer](https://arxiv.org/abs/1706.03762)⁠ based language model (LLM) discussed in the [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) repository from the excellent ["LLMs from Scratch"](https://www.manning.com/books/build-a-large-language-model-from-scratch) book by [Sebastian Raschka](https://sebastianraschka.com/).

The aim is to not only reproduce the code but to  understand the underlying mechanics and be able to modify and extend it.

---

## 🗺️ The Roadmap at a Glance

```
 [Chapter 2: Data Pipeline] ──> [Chapter 3: Attention Mechanics] ──> [Chapter 4: GPT Architecture]
                                                                               │
 [Chapters 6 & 7: Finetuning] <── [Chapter 5: Pretraining & Gen] <─────────────┘
```

---

## 📑 Chapter-by-Chapter Execution Plan

### 🟩 Step 1 — Chapter 2: The Data Pipeline & Tokenization
*Building the entry point for raw text data.*

*   **Core Architectural Concepts:**
    *   **Byte-Pair Encoding (BPE):** How subword tokenization solves the Out-Of-Vocabulary (OOV) problem using the `tiktoken` library.
    *   **Vocabulary Mapping:** Transforming text tokens into discrete integer IDs.
    *   **Special Tokens:** Defining constraints with tokens like `<|endoftext|>` and `<|unk|>`.
    *   **Sliding Window Mechanism:** Chunking long, continuous texts into training samples of fixed context size ($T$).
*   **Implementation Mechanics:**
    *   Designing a custom PyTorch `Dataset` to wrap around token sequences.
    *   Configuring a PyTorch `DataLoader` to yield aligned inputs ($x$) and targets ($y$) shifted by 1 token ($y_t = x_{t+1}$).
*   **Elaboration Focus:**
    *   Deep dive into the **Token Embedding Matrix** ($W_e$) acting as a differentiable lookup table.
    *   Implementing and justifying **Learned Positional Embeddings** ($W_p$) to inject sequence order into the transformer block, resulting in the initial vector representation: 
        $$X_0 = X_{token} + X_{position}$$
*   **"Code Notebook":**
    *   The concepts are implemented and elaborated on in [this notebook](./01-chapter02.ipynb).

---

### 🟦 Step 2 — Chapter 3: Deep Dive into Attention Mechanics
*The mathematical heart of the Transformer architecture.*

*   **Core Architectural Concepts:**
    *   The structural intuition behind **Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)**.
    *   **Scaled Dot-Product Attention:** Why dividing by $\sqrt{d_k}$ is mathematically necessary to prevent vanishing gradients in the softmax function.
    *   **Causal Masking:** Constructing a lower-triangular matrix of negative infinities ($-\infty$) to maintain the autoregressive property (preventing the model from looking ahead).
*   **Implementation Mechanics:**
    *   Writing the exact raw tensor matrix multiplication pipeline:
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
*   **Elaboration Focus:**
    *   Scaling up from **Single-Head Attention** to **Multi-Head Attention (MHA)**.
    *   Efficient tensor transposition configurations using `.view()` and `.transpose()` to split the hidden dimension ($d_{in}$) into $h$ heads of size $d_k$ without cloning memory loops.
    *   Comparing the baseline scratch matrix layout with PyTorch's native optimized `flash_attention` backend.
*   **"What Happens If?" Challenges to Run:**
    *   *What happens if we change the scaling factor from $\sqrt{d_k}$ to $1$ or $d_k$?*
    *   *How does the causal attention matrix look visually when processing text vs. generating text?*

---

### 🟨 Step 3 — Chapter 4: Assembling the GPT Architecture
*Putting the individual puzzle pieces together into a unified network.*

*   **Core Architectural Concepts:**
    *   **Layer Normalization (LayerNorm):** Stabilizing activations across the feature dimension.
    *   **Pre-LN vs. Post-LN:** Why modern architectures prefer Pre-LN (LayerNorm before the attention block) for smoother gradient flow in deep networks.
    *   **Residual Streams (Shortcut Connections):** Mitigating the vanishing gradient problem by passing the input identity directly forward ($x + 	ext{SubLayer}(x)$).
    *   **GELU Activation:** Utilizing the Gaussian Error Linear Unit function for non-linear feature transformation instead of standard ReLU.
*   **Implementation Mechanics:**
    *   Building the full `TransformerBlock` combining MHA and a multi-layer Feed-Forward Network (`Linear -> GELU -> Linear`).
    *   Orchestrating the top-level `GPTModel` class integrating inputs, stacked transformer layers, and the final output Linear projection head.
*   **Elaboration Focus:**
    *   Rigorous **Tensor Shape Tracking Matrix** mapping inputs from `[Batch Size, Seq Length]` through every layer boundary to the output logits of shape `[Batch Size, Seq Length, Vocab Size]`.
*   **"What Happens If?" Challenges to Run:**
    *   *What happens if we disable residual connections during initialization? How does it impact forward pass outputs?*
    *   *How many parameters are in our model? Let's write a tracker function to count embedding vs. attention vs. linear head parameters.*

---

### 🟥 Step 4 — Chapter 5: Pretraining & Generative Inference
*Teaching the model to write by exposing it to unlabeled data.*

*   **Core Architectural Concepts:**
    *   **Cross-Entropy Loss Function:** Evaluating model sequence probabilities against true tokens.
    *   **Autoregressive Generation:** Generating text token-by-token iteratively.
    *   **Decoding Strategies:** Understanding Temperature Scaling, Top-$k$, and Nucleus (Top-$p$) sampling to control creativity vs. determinism.
*   **Implementation Mechanics:**
    *   Writing a clean, boilerplate-free training loop with Validation and Training loss evaluation tracking.
    *   Building a complete text-generation loop that handles context windows and iteratively appends predicted tokens back into the prompt input context.
*   **Elaboration Focus:**
    *   Modern repository additions: Integrating performance accelerators such as Automatic Mixed Precision (`torch.amp`), weight initialization protocols, and the cutting-edge **Muon** optimizer.
    *   **Key-Value (KV) Caching Mechanics:** Reusing previously calculated $K$ and $V$ tensors to accelerate inference from $O(N^2)$ down to $O(1)$ per token generation step.
*   **"What Happens If?" Challenges to Run:**
    *   *What happens to text output when temperature approaches 0 vs. when it goes above 2.0?*
    *   *Let's benchmark text generation latency with KV Cache enabled vs. disabled.*

---

### 🟪 Step 5 — Chapters 6 & 7: Specialized Finetuning
*Steering a pretrained base model into a highly focused task or a conversation helper.*

*   **Core Architectural Concepts:**
    *   **Classification Finetuning (Chapter 6):** Replacing the language model language head with a specific target classifier head (e.g., spam detection).
    *   **Instruction Finetuning (Chapter 7):** Moving from next-token completion to answering user queries using explicit prompt formatting datasets (e.g., Alpaca templates).
    *   **Target Masking (Response Alignment):** Masking out prompt text loss evaluations so the model is only evaluated on its provided answers.
*   **Implementation Mechanics:**
    *   Modifying the final linear layers of a pretrained structural architecture weight file.
    *   Creating automated evaluation strategies (e.g., via `Ollama` or model-evaluated scoring).
*   **Elaboration Focus:**
    *   Implementing **Parameter-Efficient Finetuning (PEFT)** via **LoRA (Low-Rank Adaptation)** from scratch to drastically optimize trainable weights via low-rank decomposition ($W_0 + \Delta W$, where $\Delta W = B 	imes A$).
*   **"What Happens If?" Challenges to Run:**
    *   *What happens if we finetune all layers vs. only freezing the base model and finetuning the classification head?*
    *   *How many parameters do we save from updating when training with a LoRA rank of r=8 vs. full finetuning?*

---
