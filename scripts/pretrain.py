import argparse
import os
import yaml
from transformers import AutoTokenizer

from torch_linguist.utils.seed import set_seed
from torch_linguist.utils.logging import get_logger
from torch_linguist.utils.hub import ensure_hf_login_hint, find_latest_checkpoint
from torch_linguist.data.hf_datasets import load_hf_dataset, load_local_text_dataset
from torch_linguist.data.packing import (
    tokenize_dataset,
    take_n_if_needed,
    group_texts_non_streaming,
    streaming_block_pack,
)
from torch_linguist.modeling.build_model import build_causal_lm
from torch_linguist.training.trainer import build_trainer, compute_perplexity

logger = get_logger()

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def maybe_enable_wandb(train_cfg: dict):
    if train_cfg.get("report_to", "none") == "wandb":
        # Requires `pip install wandb` and `wandb login`
        os.environ.setdefault("WANDB_PROJECT", "torch-linguist")
        if train_cfg.get("run_name"):
            os.environ.setdefault("WANDB_RUN_NAME", str(train_cfg["run_name"]))

def load_tokenizer(tok_cfg: dict):
    if tok_cfg.get("tokenizer_dir"):
        tok = AutoTokenizer.from_pretrained(tok_cfg["tokenizer_dir"])
    else:
        tok = AutoTokenizer.from_pretrained(tok_cfg["pretrained_name"])

    if tok_cfg.get("pad_to_eos_if_missing", True) and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def resolve_resume(train_cfg: dict) -> str | bool | None:
    val = train_cfg.get("resume_from_checkpoint", None)
    if val is None or val == "null":
        return None
    if isinstance(val, str) and val.lower() == "auto":
        latest = find_latest_checkpoint(train_cfg["output_dir"])
        return latest if latest else None
    return val  # explicit path

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    hub_cfg = cfg.get("hub", {})

    maybe_enable_wandb(train_cfg)
    ensure_hf_login_hint(bool(hub_cfg.get("push_to_hub", False)))

    tokenizer = load_tokenizer(tok_cfg)

    # 1) Load dataset
    streaming = bool(data_cfg.get("streaming", False))
    source = data_cfg.get("source", "hf_dataset")

    if source == "hf_dataset":
        ds = load_hf_dataset(
            name=data_cfg["dataset_name"],
            config=data_cfg.get("dataset_config"),
            streaming=streaming,
        )
        text_col = data_cfg.get("text_column", "text")
    elif source == "local_text":
        lf = data_cfg["local_files"]
        ds = load_local_text_dataset(
            train_path=lf["train"],
            validation_path=lf.get("validation"),
            test_path=lf.get("test"),
            streaming=streaming,
        )
        text_col = "text"
    else:
        raise ValueError(f"Unknown data.source: {source}")

    block_size = int(data_cfg.get("block_size", 512))

    # 2) Tokenize
    tokenized = tokenize_dataset(ds, tokenizer, text_column=text_col)

    # 3) Optional take N samples
    tokenized_train = take_n_if_needed(tokenized["train"], data_cfg.get("max_train_samples"))
    tokenized_eval = tokenized.get("validation", None)
    if tokenized_eval is not None:
        tokenized_eval = take_n_if_needed(tokenized_eval, data_cfg.get("max_eval_samples"))

    # 4) Pack
    if streaming:
        train_lm = streaming_block_pack(tokenized_train, block_size=block_size)
        eval_lm = streaming_block_pack(tokenized_eval, block_size=block_size) if tokenized_eval is not None else None
    else:
        # non-streaming: pack per split (important: packing changes length)
        packed_train = group_texts_non_streaming(tokenized_train, block_size=block_size)
        train_lm = packed_train

        eval_lm = None
        if tokenized_eval is not None:
            packed_eval = group_texts_non_streaming(tokenized_eval, block_size=block_size)
            eval_lm = packed_eval

    # 5) Model
    model = build_causal_lm(
        model_cfg=model_cfg,
        vocab_size=len(tokenizer),
        block_size=block_size,
    )

    # 6) Trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_lm,
        eval_ds=eval_lm,
        train_cfg=train_cfg,
        hub_cfg=hub_cfg,
    )

    # 7) Train (resume)
    resume = resolve_resume(train_cfg)
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
    else:
        logger.info("Starting training from scratch")
    trainer.train(resume_from_checkpoint=resume)

    # 8) Evaluate
    if eval_lm is not None:
        metrics = trainer.evaluate()
        ppl = compute_perplexity(metrics.get("eval_loss"))
        logger.info(f"eval_loss={metrics.get('eval_loss'):.4f} perplexity={ppl:.2f}")

    # 9) Push to hub at end
    if hub_cfg.get("push_to_hub", False):
        logger.info("Pushing to Hugging Face Hub...")
        trainer.push_to_hub()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pretrain_gpt2_small.yaml")
    args = ap.parse_args()
    main(args.config)
