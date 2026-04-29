import json
import random
from datasets import load_dataset

OUTPUT_FILE = "questions_mmlu_100.jsonl"
SAMPLE_SIZE = 100
RANDOM_SEED = 42

LETTERS = "ABCD"

# Normal MMLU: 4 seçenekli, bizim mevcut benchmark koduna uygun.
dataset = load_dataset("cais/mmlu", "all", split="test")

rows = list(dataset)

clean_rows = []

for row in rows:
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer")

    if not question or not choices:
        continue

    if len(choices) != 4:
        continue

    if not isinstance(answer, int):
        continue

    if answer < 0 or answer > 3:
        continue

    clean_rows.append(row)

random.seed(RANDOM_SEED)
sample = random.sample(clean_rows, SAMPLE_SIZE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, row in enumerate(sample, start=1):
        choices = row["choices"]

        option_text = "\n".join(
            f"{LETTERS[i]}) {choices[i]}" for i in range(4)
        )

        question = f"{row['question']}\n{option_text}"

        item = {
            "id": idx,
            "source": "MMLU",
            "category": row.get("subject", "unknown"),
            "difficulty": "medium-hard",
            "question": question,
            "answer": LETTERS[row["answer"]],
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"{OUTPUT_FILE} oluşturuldu. Toplam soru: {SAMPLE_SIZE}")