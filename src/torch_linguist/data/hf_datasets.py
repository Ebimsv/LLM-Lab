from __future__ import annotations
from datasets import load_dataset, DatasetDict

def load_hf_dataset(name: str, config: str | None, streaming: bool) -> DatasetDict:
    return load_dataset(name, config, streaming=streaming)

def load_local_text_dataset(
    train_path: str,
    validation_path: str | None = None,
    test_path: str | None = None,
    streaming: bool = False,
) -> DatasetDict:
    data_files = {"train": train_path}
    if validation_path:
        data_files["validation"] = validation_path
    if test_path:
        data_files["test"] = test_path

    # load_dataset("text") yields a dataset with column name "text"
    return load_dataset("text", data_files=data_files, streaming=streaming)
