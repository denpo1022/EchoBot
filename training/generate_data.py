"""
Phase 1 & 2: Generate CAI training data for EchoBot
- Generate initial responses from seed prompts
- Critique each response against the constitution
- Revise based on critique
- Save preference pairs for DPO training
"""

import json
import os
import time
from pathlib import Path

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CONSTITUTION = Path("constitution.md").read_text(encoding="utf-8")

ECHOBOT_SYSTEM = """你是一位带一点猫娘气质的助手。
- 自称可以是"我"或"本喵"，但不要每句话都重复自称。
- 语气轻快、亲近、俏皮，偶尔在句尾自然地加"喵"。
- 不要过度卖萌，不要连续堆叠语气词，不要影响信息清晰度。
- 遇到严肃问题时，先保证准确和有条理，再保留一点温柔的角色感。
- 默认使用简洁中文回复。"""

CRITIQUE_PROMPT = """You are evaluating an AI assistant's response against a constitution.

Constitution:
{constitution}

User message: {prompt}

Assistant response to evaluate:
{response}

Critique this response against each constitutional principle. Be specific about what is wrong or could be improved. If the response is already perfect, say so."""

REVISION_PROMPT = """You are improving an AI assistant's response based on a critique.

Constitution:
{constitution}

User message: {prompt}

Original response:
{response}

Critique:
{critique}

Now write an improved response that addresses all issues raised in the critique while maintaining the EchoBot cat-girl persona. Output only the improved response, nothing else."""


def chat(system: str, user: str, model: str = "llama-3.1-70b-versatile") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def generate_initial_response(prompt: str) -> str:
    return chat(ECHOBOT_SYSTEM, prompt)


def critique_response(prompt: str, response: str) -> str:
    return chat(
        "You are a strict AI safety and quality evaluator.",
        CRITIQUE_PROMPT.format(
            constitution=CONSTITUTION,
            prompt=prompt,
            response=response,
        ),
    )


def revise_response(prompt: str, response: str, critique: str) -> str:
    return chat(
        ECHOBOT_SYSTEM,
        REVISION_PROMPT.format(
            constitution=CONSTITUTION,
            prompt=prompt,
            response=response,
            critique=critique,
        ),
    )


def process_prompt(prompt: str) -> dict | None:
    try:
        print(f"\n[PROMPT] {prompt[:60]}...")

        initial = generate_initial_response(prompt)
        print(f"[INITIAL] {initial[:80]}...")
        time.sleep(2)  # respect rate limits

        critique = critique_response(prompt, initial)
        print(f"[CRITIQUE] {critique[:80]}...")
        time.sleep(2)

        revised = revise_response(prompt, initial, critique)
        print(f"[REVISED] {revised[:80]}...")
        time.sleep(2)

        return {
            "prompt": prompt,
            "chosen": revised,
            "rejected": initial,
            "critique": critique,
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def main():
    seed_file = Path("data/seed_prompts.jsonl")
    output_file = Path("data/dpo_pairs.jsonl")

    prompts = [json.loads(line)["prompt"] for line in seed_file.read_text().splitlines() if line.strip()]
    print(f"Processing {len(prompts)} prompts...")

    existing = set()
    if output_file.exists():
        for line in output_file.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)["prompt"])
        print(f"Skipping {len(existing)} already processed prompts.")

    with output_file.open("a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            if prompt in existing:
                continue

            result = process_prompt(prompt)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{len(prompts)}] Saved.")

    print(f"\nDone! {output_file} contains {sum(1 for _ in output_file.open())} pairs.")


if __name__ == "__main__":
    main()
