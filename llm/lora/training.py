"""LoRA fine-tuning utilities for the paper's stage-1 model (Eq. 13-16)."""
import json
import math
from pathlib import Path

import config

IGNORE_INDEX = -100
ROLE_MARKERS = {"system": "[SYSTEM]", "user": "[USER]", "assistant": "[ASSISTANT]"}


def render_messages(messages):
    """Render training/inference messages with the paper's role-marker format."""
    rendered = []
    for message in messages:
        role = message.get("role", "").lower()
        if role not in ROLE_MARKERS:
            raise ValueError(f"Unsupported LoRA chat role: {role!r}")
        rendered.append(f"{ROLE_MARKERS[role]}\n{message.get('content', '').strip()}\n")
    return "".join(rendered)


def render_training_row(row):
    required = ("system", "user", "assistant")
    missing = [key for key in required if not isinstance(row.get(key), str) or not row[key].strip()]
    if missing:
        raise ValueError(f"LoRA training row is missing non-empty text fields: {', '.join(missing)}")
    prefix = render_messages([
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["user"]},
    ]) + f"{ROLE_MARKERS['assistant']}\n"
    response = row["assistant"].strip()
    return prefix, prefix + response


def _input_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def maximum_sequence_length(rows, tokenizer, maximum=512):
    """Paper Eq. (13), capped by the paper's 512-token maximum."""
    if not rows:
        raise ValueError("LoRA fine-tuning needs at least one training example.")
    longest = max(len(_input_ids(tokenizer, render_training_row(row)[1])) for row in rows)
    return max(128, min(512, maximum, math.ceil(1.2 * longest)))


def tokenize_training_row(row, tokenizer, max_length):
    """Tokenize a row and mask the system/user/prompt tokens from the causal-LM loss."""
    prefix, full_text = render_training_row(row)
    prefix_ids = _input_ids(tokenizer, prefix)
    input_ids = _input_ids(tokenizer, full_text)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not input_ids or input_ids[-1] != eos_id):
        input_ids.append(eos_id)

    input_ids = input_ids[:max_length]
    prompt_length = min(len(prefix_ids), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("LoRA sequence truncation removed the entire assistant response.")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_training_rows(path: Path = config.LORA_TRAINING_FILE):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"LoRA training file contains no examples: {path}")
    for index, row in enumerate(rows):
        try:
            render_training_row(row)
        except ValueError as exc:
            raise ValueError(f"Invalid LoRA training row {index + 1} in {path}: {exc}") from exc
    return rows


def train_lora(training_file: Path = config.LORA_TRAINING_FILE, output_dir: Path = config.LORA_ADAPTER_DIR):
    """Train and save the paper-configured LoRA adapter; requires a CUDA bf16-capable GPU."""
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires torch, transformers, accelerate, and peft. "
            "Install the optional LoRA dependencies from requirements.txt."
        ) from exc

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Paper-configured LoRA training requires a CUDA GPU with bfloat16 support. "
            "No training was started."
        )

    training_file, output_dir = Path(training_file), Path(output_dir)
    rows = load_training_rows(training_file)
    tokenizer = AutoTokenizer.from_pretrained(config.LORA_BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = maximum_sequence_length(rows, tokenizer, config.LORA_MAX_LENGTH)
    dataset_rows = [tokenize_training_row(row, tokenizer, max_length) for row in rows]

    model = AutoModelForCausalLM.from_pretrained(
        config.LORA_BASE_MODEL,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )

    class TrainingRows(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset_rows)

        def __getitem__(self, index):
            return dataset_rows[index]

    class PaddingCollator:
        def __call__(self, features):
            width = max(len(row["input_ids"]) for row in features)
            pad_id = tokenizer.pad_token_id
            return {
                "input_ids": torch.tensor(
                    [row["input_ids"] + [pad_id] * (width - len(row["input_ids"])) for row in features],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [row["attention_mask"] + [0] * (width - len(row["attention_mask"])) for row in features],
                    dtype=torch.long,
                ),
                "labels": torch.tensor(
                    [row["labels"] + [IGNORE_INDEX] * (width - len(row["labels"])) for row in features],
                    dtype=torch.long,
                ),
            }

    checkpoint_dir = output_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.LORA_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.LORA_GRAD_ACCUMULATION,
        learning_rate=config.LORA_LEARNING_RATE,
        weight_decay=config.LORA_WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_steps=config.LORA_WARMUP_STEPS,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=TrainingRows(), data_collator=PaddingCollator())
    result = trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": config.LORA_BASE_MODEL,
        "training_file": str(training_file),
        "examples": len(dataset_rows),
        "max_sequence_length": max_length,
        "train_loss": result.metrics.get("train_loss"),
        "lora": {
            "rank": config.LORA_RANK,
            "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT,
            "target_modules": ["q_proj", "v_proj"],
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.LORA_LEARNING_RATE,
            "weight_decay": config.LORA_WEIGHT_DECAY,
            "scheduler": "cosine",
            "warmup_steps": config.LORA_WARMUP_STEPS,
        },
        "epochs": config.LORA_EPOCHS,
        "batch_size": 1,
        "gradient_accumulation_steps": config.LORA_GRAD_ACCUMULATION,
        "bf16": True,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata
