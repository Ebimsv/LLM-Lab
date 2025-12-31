import argparse
import yaml
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def iter_text_lines(files):
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    tok_cfg = cfg["tokenizer"]
    files = cfg["data"]["files"]

    out_dir = Path(tok_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=bool(tok_cfg.get("add_prefix_space", True)))

    trainer = BpeTrainer(
        vocab_size=int(tok_cfg["vocab_size"]),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        special_tokens=list(tok_cfg["special_tokens"]),
    )

    tokenizer.train_from_iterator(iter_text_lines(files), trainer=trainer)
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
