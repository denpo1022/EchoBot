# EchoBot Constitutional AI Training

## Overview

This pipeline trains EchoBot using Constitutional AI (CAI) — the model critiques and revises its own outputs, producing preference pairs for DPO fine-tuning. No human labelers needed.

## Pipeline

```
Seed Prompts → Initial Response → Critique → Revised Response → DPO Pairs → Fine-tuned Model
```

## Setup

```bash
pip install groq unsloth trl datasets transformers accelerate bitsandbytes
```

Add your Groq key to `.env`:
```bash
GROQ_API_KEY=gsk_your_key_here
```

## Step 1 — Generate Training Data

```bash
cd training
export $(cat ../.env | grep GROQ_API_KEY)
python generate_data.py
```

This generates `data/dpo_pairs.jsonl` with ~30 critique-revised preference pairs.
Add more prompts to `data/seed_prompts.jsonl` for better coverage (aim for 500+).

## Step 2 — DPO Fine-tuning

```bash
python train_dpo.py
```

- Model: `Qwen2.5-7B-Instruct` (4-bit quantized)
- VRAM: ~6GB (fits RTX 3060 Ti)
- Time: ~2-4 hours

Output saved to `models/echobot-cai/`

## Step 3 — Evaluate

```bash
python evaluate.py
```

## Files

| File | Purpose |
|------|---------|
| `constitution.md` | EchoBot's principles for self-critique |
| `generate_data.py` | Phase 1 & 2: generate DPO preference pairs via Groq |
| `train_dpo.py` | Phase 3: DPO fine-tuning with unsloth |
| `evaluate.py` | Compare fine-tuned model responses |
| `data/seed_prompts.jsonl` | Input prompts (add more for better training) |
| `data/dpo_pairs.jsonl` | Generated preference pairs (auto-created) |
| `models/echobot-cai/` | Fine-tuned model output |
