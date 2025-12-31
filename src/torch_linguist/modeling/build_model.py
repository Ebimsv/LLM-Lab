from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM

def build_causal_lm(model_cfg: dict, vocab_size: int, block_size: int):
    kind = model_cfg.get("kind", "gpt2_from_scratch")

    if kind == "gpt2_from_scratch":
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=block_size,
            n_ctx=block_size,
            n_layer=int(model_cfg.get("n_layer", 8)),
            n_head=int(model_cfg.get("n_head", 8)),
            n_embd=int(model_cfg.get("n_embd", 512)),
            resid_pdrop=float(model_cfg.get("dropout", 0.0)),
            embd_pdrop=float(model_cfg.get("dropout", 0.0)),
            attn_pdrop=float(model_cfg.get("dropout", 0.0)),
        )
        return GPT2LMHeadModel(config)

    if kind == "pretrained":
        name = model_cfg["pretrained_name"]
        return AutoModelForCausalLM.from_pretrained(name)

    raise ValueError(f"Unknown model kind: {kind}")
