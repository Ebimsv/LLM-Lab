# Torch-Linguist (Hugging Face Edition)

Modern, modular **causal language model (CLM) pretraining** baseline using the **Hugging Face ecosystem**.

This repository is a modernization of an older **torchtext-based** language modeling project into current best practices using:

* **🤗 Transformers** (models, Trainer, checkpoints)
* **🤗 Datasets** (HF datasets + local text + streaming)
* **🤗 Tokenizers** (optional BPE tokenizer training)
* Optional: **Weights & Biases** logging and **Hugging Face Hub** publishing

The focus is **causal LLMs** (next-token prediction). The code is designed to be simple now and easy to extend later to:

* Supervised fine-tuning (SFT)
* Instruction tuning
* Preference tuning (DPO/ORPO, etc.)

---

## Features

### Training

* ✅ **Causal LM pretraining** (labels = input_ids)
* ✅ Standard **tokenization → packing (block_size) → Trainer**
* ✅ **Checkpointing** and **resume from latest checkpoint**
* ✅ Mixed precision (**fp16/bf16**) (configurable)
* ✅ Optional **W&B logging**
* ✅ Optional **push_to_hub** (upload checkpoints/model)

### Data

* ✅ Train on **HF datasets** (e.g., WikiText)
* ✅ Train on **local text files** (`load_dataset("text", data_files=...)`)
* ✅ **Streaming mode** for large datasets
* ✅ Optional `max_train_samples` / `max_eval_samples` for fast tutorials

### Tokenizer

* ✅ Use pretrained tokenizer (e.g., GPT-2) **or**
* ✅ Train a **BPE tokenizer** from your local corpus and save as a HF tokenizer folder

---

## Project Layout

```txt
Torch-Linguist/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ pretrain_gpt2_small.yaml
│  └─ tokenizer_bpe.yaml
├─ scripts/
│  ├─ pretrain.py
│  ├─ train_tokenizer.py
│  └─ generate.py
└─ src/torch_linguist/
   ├─ data/          # dataset loading + packing
   ├─ modeling/      # model building (from scratch or pretrained)
   ├─ training/      # Trainer wiring + perplexity helpers
   └─ utils/         # seed, logging, hub helpers
```

---

## How the Data Flows (CLM Pretraining)

The training pipeline follows the standard causal LM pretraining recipe:

1. **Load dataset**

   * HF dataset: `load_dataset(dataset_name, dataset_config)`
   * Local text: `load_dataset("text", data_files=...)`

2. **Tokenize**

   * Convert raw text into token IDs: `input_ids`

3. **Pack into fixed blocks**

   * Concatenate token streams
   * Split into fixed-length sequences: `block_size` (e.g., 512)
   * Create labels: `labels = input_ids`

4. **Train**

   * Use `Trainer` + `DataCollatorForLanguageModeling(mlm=False)`
   * Save checkpoints and optionally resume / push to hub

---

## Installation

```bash
pip install -r requirements.txt
```

Optional (for Weights & Biases logging):

```bash
pip install wandb
```

---

## Quickstart: Pretraining on WikiText-2

This is the recommended “tutorial default” dataset: small, fast, and good for learning.

```bash
PYTHONPATH=src python scripts/pretrain.py --config configs/pretrain_gpt2_small.yaml
```

Checkpoints will be written to:

```
runs/pretrain_gpt2_small/
```

---

## Generate Text from a Checkpoint

```bash
PYTHONPATH=src python scripts/generate.py \
  --model_dir runs/pretrain_gpt2_small/checkpoint-500 \
  --prompt "Hello my name is"
```

---

## Tokenizer Options

### Option A (Simplest): Use a Pretrained Tokenizer

In `configs/pretrain_gpt2_small.yaml`:

```yaml
tokenizer:
  pretrained_name: gpt2
  pad_to_eos_if_missing: true
```

### Option B: Train Your Own BPE Tokenizer

1. Put your corpus files in `data/raw/` (or adjust paths in config)

2. Train tokenizer:

```bash
PYTHONPATH=src python scripts/train_tokenizer.py --config configs/tokenizer_bpe.yaml
```

3. Update `configs/pretrain_gpt2_small.yaml` to use the tokenizer folder:

```yaml
tokenizer:
  tokenizer_dir: artifacts/tokenizer_bpe
  pad_to_eos_if_missing: true
```

Then train:

```bash
PYTHONPATH=src python scripts/pretrain.py --config configs/pretrain_gpt2_small.yaml
```

---

## Training on Local Text Files

Set in `configs/pretrain_gpt2_small.yaml`:

```yaml
data:
  source: local_text
  local_files:
    train: data/raw/wiki.train.tokens
    validation: data/raw/wiki.valid.tokens
    test: data/raw/wiki.test.tokens
```

Then run:

```bash
PYTHONPATH=src python scripts/pretrain.py --config configs/pretrain_gpt2_small.yaml
```

---

## Streaming Mode (Large Datasets)

For very large datasets (web-scale text), enable streaming:

```yaml
data:
  streaming: true
```

This uses a streaming-friendly block packing approach.

---

## Resume Training

In `configs/pretrain_gpt2_small.yaml`:

```yaml
train:
  resume_from_checkpoint: auto
```

* `auto` resumes from the most recent `checkpoint-*` in `output_dir`.
* You can also set an explicit checkpoint path:

  ```yaml
  resume_from_checkpoint: runs/pretrain_gpt2_small/checkpoint-1000
  ```

---

## Logging with Weights & Biases (Optional)

1. Install and login:

```bash
pip install wandb
wandb login
```

2. Enable in config:

```yaml
train:
  report_to: wandb
  run_name: "wikitext2-gpt2-small"
```

---

## Push to Hugging Face Hub (Optional)

1. Login once:

```bash
huggingface-cli login
```

2. Enable in config:

```yaml
hub:
  push_to_hub: true
  repo_id: "YOUR_USERNAME/torch-linguist-pretrain"
  private: true
```

During training, checkpoints can be uploaded depending on `hub_strategy`, and at the end the script runs `trainer.push_to_hub()`.

---

## Configuration Guide (What to Tune First)

### `data.block_size`

* 256/512 for quick experiments
* 1024+ for stronger modeling (more VRAM)

### Batch size vs gradient accumulation

If VRAM is limited:

* keep `per_device_train_batch_size` small
* increase `gradient_accumulation_steps`

### Model size

In `model` config:

* `n_layer`, `n_head`, `n_embd` control model capacity
* start small to validate the pipeline, then scale up

---

## Roadmap: Fine-tuning & Instruction Tuning

This repo is structured so adding later stages is straightforward:

### 1) Supervised Fine-Tuning (SFT)

Add:

* `scripts/finetune_sft.py`
* `src/torch_linguist/data/sft.py` (format: prompt → response)
* Use a data collator that masks prompt tokens if needed.

### 2) Instruction / Preference Tuning (Optional)

Add:

* `scripts/train_dpo.py` (or ORPO)
* dataset formatting for (prompt, chosen, rejected)
* a trainer from `trl` (HF TRL library)

---

## License

Add your license here (MIT / Apache-2.0 / etc.).

---

## Acknowledgements

* Hugging Face Transformers, Datasets, Tokenizers, Accelerate
* WikiText dataset authors (if using wikitext)
