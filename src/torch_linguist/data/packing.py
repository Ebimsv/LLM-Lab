from __future__ import annotations

def tokenize_dataset(ds, tokenizer, text_column: str):
    def _tok(batch):
        return tokenizer(batch[text_column], return_special_tokens_mask=False)
    return ds.map(_tok, batched=True, remove_columns=[text_column])

def take_n_if_needed(ds, n: int | None):
    """
    Non-streaming: ds.select(range(n))
    Streaming: ds.take(n) -> IterableDataset
    """
    if n is None:
        return ds
    if hasattr(ds, "select"):
        return ds.select(range(min(n, len(ds))))
    return ds.take(n)

def group_texts_non_streaming(tokenized_ds, block_size: int):
    """
    Standard CLM packing for normal (non-streaming) datasets.
    """
    def _group(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_ds.map(_group, batched=True)

def streaming_block_pack(iterable_tokenized_ds, block_size: int):
    """
    Packing for streaming datasets.
    Yields {input_ids, labels} blocks of size block_size.
    """
    # datasets IterableDataset supports .from_generator in recent versions.
    cls = iterable_tokenized_ds.__class__

    def gen():
        buffer = []
        for item in iterable_tokenized_ds:
            buffer.extend(item["input_ids"])
            while len(buffer) >= block_size:
                block = buffer[:block_size]
                buffer = buffer[block_size:]
                yield {"input_ids": block, "labels": block.copy()}

    return cls.from_generator(gen)
