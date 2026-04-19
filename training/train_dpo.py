"""
Phase 3: DPO fine-tuning on EchoBot preference pairs
Requires: unsloth, trl, datasets, transformers
GPU: RTX 3060 Ti (8GB VRAM) — uses 4-bit quantization + LoRA
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "models/echobot-cai"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = """你是一位带一点猫娘气质的助手。
- 自称可以是"我"或"本喵"，但不要每句话都重复自称。
- 语气轻快、亲近、俏皮，偶尔在句尾自然地加"喵"。
- 不要过度卖萌，不要连续堆叠语气词，不要影响信息清晰度。
- 遇到严肃问题时，先保证准确和有条理，再保留一点温柔的角色感。
- 默认使用简洁中文回复。"""


def load_dpo_dataset(path: str = "data/dpo_pairs.jsonl") -> Dataset:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ],
            "chosen": [{"role": "assistant", "content": item["chosen"]}],
            "rejected": [{"role": "assistant", "content": item["rejected"]}],
        })
    print(f"Loaded {len(records)} DPO pairs.")
    return Dataset.from_list(records)


def main():
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        use_gradient_checkpointing=True,
        random_state=42,
    )

    dataset = load_dpo_dataset()

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        beta=0.1,                          # DPO temperature
        max_prompt_length=512,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    main()
