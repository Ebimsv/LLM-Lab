import argparse
import yaml
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

from torch_linguist.data.hf_datasets import load_hf_dataset

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def iter_text_lines(files):
    """Iterator for local text files."""
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def iter_hf_dataset(dataset_dict, splits, text_column):
    """Iterator for HuggingFace datasets."""
    for split_name in splits:
        if split_name in dataset_dict:
            dataset = dataset_dict[split_name]
            for example in dataset:
                text = example.get(text_column, "")
                if text and text.strip():
                    yield text.strip()

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    tok_cfg = cfg["tokenizer"]
    data_cfg = cfg["data"]
    
    source = data_cfg.get("source", "local_files")
    
    # Get data iterator based on source
    if source == "hf_dataset":
        dataset_name = data_cfg["dataset_name"]
        dataset_config = data_cfg.get("dataset_config")
        text_column = data_cfg.get("text_column", "text")
        splits = data_cfg.get("splits", ["train"])
        
        print(f"Loading HuggingFace dataset: {dataset_name} (config: {dataset_config})")
        dataset_dict = load_hf_dataset(
            name=dataset_name,
            config=dataset_config,
            streaming=False  # Need full dataset for tokenizer training
        )
        data_iterator = iter_hf_dataset(dataset_dict, splits, text_column)
    elif source == "local_files":
        files = data_cfg.get("files", [])
        if not files:
            raise ValueError("data.files must be provided when source=local_files")
        print(f"Loading local files: {files}")
        data_iterator = iter_text_lines(files)
    else:
        raise ValueError(f"Unknown data.source: {source}. Must be 'hf_dataset' or 'local_files'")

    out_dir = Path(tok_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=bool(tok_cfg.get("add_prefix_space", True)))

    trainer = BpeTrainer(
        vocab_size=int(tok_cfg["vocab_size"]),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        special_tokens=list(tok_cfg["special_tokens"]),
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tok.save_pretrained(str(out_dir))

    print(f"Saved tokenizer folder: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/tokenizer_bpe.yaml")
    args = ap.parse_args()
    main(args.config)
