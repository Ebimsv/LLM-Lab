import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(model_dir: str, prompt: str, max_new_tokens: int):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    args = ap.parse_args()
    main(args.model_dir, args.prompt, args.max_new_tokens)
