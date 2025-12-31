import math
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

def compute_perplexity(eval_loss: float | None) -> float:
    if eval_loss is None:
        return float("nan")
    if eval_loss > 50:
        return float("inf")
    return float(math.exp(eval_loss))

def build_trainer(model, tokenizer, train_ds, eval_ds, train_cfg: dict, hub_cfg: dict):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    push_to_hub = bool(hub_cfg.get("push_to_hub", False))
    hub_model_id = hub_cfg.get("repo_id", None) if push_to_hub else None

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),

        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),

        logging_steps=int(train_cfg["logging_steps"]),
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),

        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),

        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name", None),

        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_num_workers=2,

        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=bool(hub_cfg.get("private", True)),
        hub_strategy=hub_cfg.get("hub_strategy", "every_save"),
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
