from __future__ import annotations
import os
from pathlib import Path

def ensure_hf_login_hint(push_to_hub: bool) -> None:
    if not push_to_hub:
        return
    # We can't verify login reliably here without calling CLI; provide a friendly hint.
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print("[INFO] Hub push enabled. Make sure you're logged in: `huggingface-cli login` "
              "or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var.")

def find_latest_checkpoint(output_dir: str) -> str | None:
    """
    Finds the most recent checkpoint-* directory under output_dir.
    """
    p = Path(output_dir)
    if not p.exists():
        return None
    cks = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not cks:
        return None
    # checkpoint-<step> sorting
    def step_num(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except Exception:
            return -1
    cks.sort(key=step_num, reverse=True)
    return str(cks[0])
