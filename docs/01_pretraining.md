# 01 — Causal Language Model Pretraining (From First Principles)

This document explains **causal language model (CLM) pretraining** step by step, using the code structure in this repository.

The goal is not only to train a model, but to **understand what is happening at each stage** and *why* modern Hugging Face–based pipelines are structured this way.

---

## What Is Causal Language Modeling?

Causal language modeling is the task of predicting the **next token** given all previous tokens.

Formally, we model:

P(xₜ | x₁, x₂, ..., xₜ₋₁)

This is the objective used by:
- GPT-2
- GPT-3 / GPT-4 (decoder-only)
- LLaMA
- Mistral
- Most modern LLMs

---

## Why “Causal” Matters

In a causal LM:

- The model **cannot see future tokens**
- Attention is **masked** so position *t* only attends to `< t`
- Training and generation use the **same objective**

This makes causal LMs:
- Easy to generate text from
- Flexible for fine-tuning and instruction tuning later

---

## High-Level Training Pipeline

Our pipeline looks like this:

```

Raw Text
↓
Tokenizer
↓
Token IDs
↓
Concatenate + Pack into Fixed Blocks
↓
(input_ids, labels)
↓
Causal Transformer
↓
Cross-Entropy Loss

````

Each step is explained below.

---

## Step 1 — Loading the Dataset

We support two modes:

### 1. Hugging Face datasets
Example:
```python
load_dataset("wikitext", "wikitext-2-raw-v1")
````

Pros:

* Clean
* Reproducible
* Great for tutorials and experiments

### 2. Local text files

Example:

```python
load_dataset("text", data_files=...)
```

Pros:

* Works with your own corpora
* Ideal for private or domain-specific data

Both modes produce a dataset with a `text` column.

---

## Step 2 — Tokenization

Raw text cannot be fed directly into a neural network.

We convert text into **tokens** using a tokenizer.

### Why Subword Tokenization?

Modern LLMs use **subword tokenization** (BPE / Unigram):

* Handles rare words gracefully
* Avoids exploding vocabulary size
* Works across languages and domains

Examples:

```
"unbelievable" → ["un", "believ", "able"]
```

In this repo you can:

* Use a pretrained tokenizer (e.g. GPT-2), or
* Train your own tokenizer on your corpus

---

## Step 3 — Token IDs

After tokenization, text becomes sequences of integers:

```
"Hello world"
→ [15496, 995]
```

At this stage:

* Sequence lengths vary
* We **do not** yet have fixed-size inputs

---

## Step 4 — Packing into Fixed Blocks

Transformers expect **fixed-length sequences**.

We therefore:

1. Concatenate token streams
2. Split into blocks of size `block_size`

Example (`block_size = 8`):

```
Tokens:
[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

Blocks:
[10, 11, 12, 13, 14, 15, 16, 17]
[18, 19, 20, ...]  → (dropped or padded)
```

This is standard practice in LLM pretraining.

---

## Step 5 — Labels for Causal LM

For causal language modeling:

```
labels = input_ids
```

Why?

Because the model internally shifts labels by one position and applies a **causal attention mask**.

So the model learns:

```
input_ids[t-1] → predict input_ids[t]
```

This is handled automatically by:

```python
DataCollatorForLanguageModeling(mlm=False)
```

---

## Step 6 — The Model

We use a **decoder-only Transformer**, typically GPT-style.

Key components:

* Token embeddings
* Positional embeddings
* Masked self-attention
* Feed-forward layers
* Layer normalization

In this repo you can:

* Train **from scratch** (GPT-2-style config), or
* Start from a **pretrained model**

---

## Step 7 — Training with Hugging Face Trainer

The `Trainer` handles:

* Forward / backward passes
* Gradient accumulation
* Mixed precision (fp16 / bf16)
* Checkpoint saving
* Evaluation
* Resuming from checkpoints

This removes a lot of fragile custom training code and makes experiments reproducible.

---

## Step 8 — Perplexity

Perplexity is the standard evaluation metric for language models.

```
perplexity = exp(cross_entropy_loss)
```

Lower perplexity means:

* The model is less “surprised” by the data
* Better next-token prediction

---

## Step 9 — Text Generation

Once trained, generation works by:

1. Providing a prompt
2. Repeatedly sampling the next token
3. Appending it to the input

Sampling parameters:

* `temperature`
* `top_p`
* `top_k`

Generation uses **the same causal objective** as training.

---

## Why This Design Scales Well

This pipeline:

* Matches how real LLMs are trained
* Is memory-efficient
* Works with streaming data
* Is easy to extend to:

  * Fine-tuning
  * Instruction tuning
  * Preference learning

---

## What Comes Next?

In the next documents, we will cover:

* Supervised fine-tuning (SFT)
* Instruction formatting
* Prompt masking strategies
* Preference tuning (DPO / ORPO)