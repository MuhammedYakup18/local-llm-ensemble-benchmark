import json
import re
import time
from collections import Counter

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

RUN_NAME = "v2_three_systems_100"

PRIMARY_MODEL = "gemma3:12b"

MODELS = [
    "gemma3:12b",
    "llama3.1:8b",
    "mistral-nemo:12b",
]

MAX_QUESTIONS = 100
VALID_CHOICES = ["A", "B", "C", "D"]


def ask_ollama(model: str, prompt: str, temperature: float = 0, timeout: int = 900) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 1,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


def extract_choice(text: str) -> str:
    if not text:
        return "?"

    text_upper = text.strip().upper()

    patterns = [
        r"ANSWER\s*[:\-]?\s*([ABCD])",
        r"FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])",
        r"FINAL\s*[:\-]?\s*([ABCD])",
        r"CHOICE\s*[:\-]?\s*([ABCD])",
        r"CEVAP\s*[:\-]?\s*([ABCD])",
        r"YANIT\s*[:\-]?\s*([ABCD])",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            return matches[-1]

    matches = re.findall(r"\b([ABCD])\b", text_upper)
    if matches:
        return matches[-1]

    return "?"


def extract_confidence(text: str) -> int:
    if not text:
        return 50

    text_upper = text.strip().upper()

    patterns = [
        r"CONFIDENCE\s*[:\-]?\s*(\d{1,3})",
        r"CONFIDENCE\s*SCORE\s*[:\-]?\s*(\d{1,3})",
        r"G[ÜU]VEN\s*[:\-]?\s*(\d{1,3})",
        r"(\d{1,3})\s*/\s*100",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            value = int(matches[-1])
            return max(0, min(100, value))

    return 50


def build_initial_prompt(question: str) -> str:
    return f"""
You are solving a multiple-choice benchmark question.

Rules:
- Choose exactly one option: A, B, C, or D.
- Give a brief reason.
- Do not mention any option outside A-D.
- End your response exactly in this format:

REASON: your brief reason
ANSWER: X
CONFIDENCE: 0-100

Question:
{question}
"""


def build_judge_always_prompt_v2(question: str, model_raw_answers: dict, model_choices: dict, model_confidences: dict) -> str:
    blocks = []

    for model in MODELS:
        blocks.append(f"""
--- {model} ---
Choice: {model_choices.get(model, "?")}
Self-reported confidence: {model_confidences.get(model, 50)}
Full reasoning:
{model_raw_answers.get(model, "")}
""")

    return f"""
You are a strict multiple-choice benchmark judge.

Question:
{question}

Independent model answers:
{''.join(blocks)}

Important observations from prior benchmark failures:
- Majority vote is often wrong when two weaker models repeat the same mistake.
- Self-reported confidence is often poorly calibrated; a model can be 100% confident and still wrong.
- A minority answer can be correct if its reasoning follows the question more directly.
- Do not reward long or confident explanations unless they directly support the option.
- Do not assume the primary model is correct; check the question itself.

Judging protocol:
1. First solve the question independently from the original text.
2. Identify the two most plausible options.
3. For each plausible option, briefly state the strongest reason it could be correct.
4. Inspect the model answers only after your own independent check.
5. Treat model votes as weak evidence, not proof.
6. Treat confidence scores as weak evidence, not proof.
7. If two models agree but their reasoning is shallow or unsupported, do not follow them automatically.
8. If one model is in the minority but gives the strongest reasoning, select the minority answer.
9. Final answer must be based on the question and reasoning quality, not vote count.

Output exactly in this format:
INDEPENDENT_TOP_TWO: X,Y
MAJORITY_RISK: explain whether the majority may be misleading
BEST_REASONING_SOURCE: model name or independent
FINAL_REASON: brief reason
ANSWER: X
CONFIDENCE: 0-100
"""


def build_gemma_final_arbiter_prompt_v2(question: str, initial_raw: dict, initial_choices: dict, initial_confidences: dict) -> str:
    blocks = []

    for model in MODELS:
        blocks.append(f"""
--- {model} ---
Initial choice: {initial_choices.get(model, "?")}
Self-reported confidence: {initial_confidences.get(model, 50)}
Full reasoning:
{initial_raw.get(model, "")}
""")

    return f"""
You are {PRIMARY_MODEL}, the primary final arbiter in a local multi-model verification system.

Question:
{question}

Independent model answers:
{''.join(blocks)}

Important observations from prior benchmark failures:
- Your own first answer is often strong, but it is not automatically correct.
- Two other models agreeing against you is not automatically proof; they can share the same mistake.
- Self-reported confidence is not reliable proof. Models have been confidently wrong.
- The best final answer is often the one with the strongest direct support from the question, not the most votes.
- If all models agree, still verify the answer once because shared errors can happen.

Final arbiter protocol:
1. Re-read the question carefully.
2. Ignore vote count at first.
3. Identify the two most plausible options.
4. State the strongest argument for your initial answer.
5. State the strongest argument against your initial answer.
6. Compare the other models' reasoning against your own reasoning.
7. If another model exposes a concrete flaw in your reasoning, change your answer.
8. If the other models only repeat an unsupported or shallow argument, keep your answer.
9. If all models agree, verify that the agreed answer follows from the question and is not a shared misconception.
10. Choose exactly one option: A, B, C, or D.

Output exactly in this format:
TOP_TWO: X,Y
ARGUMENT_FOR_INITIAL: ...
ARGUMENT_AGAINST_INITIAL: ...
MAJORITY_RISK: low/medium/high
FINAL_REASON: ...
ANSWER: X
CONFIDENCE: 0-100
"""


def build_sequential_llama_revision_prompt(question: str, llama_initial: str, gemma_initial: str) -> str:
    return f"""
Question:
{question}

You are llama3.1:8b.

Your independent initial answer was:
{llama_initial}

The primary model {PRIMARY_MODEL} initially answered:
{gemma_initial}

Task:
- Review {PRIMARY_MODEL}'s answer and reasoning.
- Decide whether to keep your initial answer or revise it.
- Explain why you kept or changed your answer.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your reason for keeping or changing
ANSWER: X
CONFIDENCE: 0-100
"""


def build_sequential_mistral_revision_prompt(
    question: str,
    mistral_initial: str,
    gemma_initial: str,
    llama_initial: str,
    llama_revised: str,
) -> str:
    return f"""
Question:
{question}

You are mistral-nemo:12b.

Your independent initial answer was:
{mistral_initial}

The primary model {PRIMARY_MODEL} initially answered:
{gemma_initial}

llama3.1:8b initially answered:
{llama_initial}

llama3.1:8b then revised/confirmed its answer:
{llama_revised}

Task:
- Review all previous answers.
- Decide whether to keep your initial answer or revise it.
- Explain why you kept or changed your answer.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your reason for keeping or changing
ANSWER: X
CONFIDENCE: 0-100
"""


def build_sequential_gemma_final_prompt(
    question: str,
    gemma_initial: str,
    llama_initial: str,
    llama_revised: str,
    mistral_initial: str,
    mistral_revised: str,
) -> str:
    return f"""
You are {PRIMARY_MODEL}, the primary final arbiter in a sequential review chain.

Question:
{question}

Your initial answer:
{gemma_initial}

llama3.1:8b initial answer:
{llama_initial}

llama3.1:8b revised answer and reason:
{llama_revised}

mistral-nemo:12b initial answer:
{mistral_initial}

mistral-nemo:12b revised answer and reason:
{mistral_revised}

Task:
- Reconsider your initial answer after reading the other models' initial and revised reasoning.
- You are not required to keep your initial answer.
- If the other models expose an error in your reasoning, revise your answer.
- If your initial reasoning is still strongest, keep it.
- Final decision must be yours only.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your final reason
ANSWER: X
CONFIDENCE: 0-100
"""


def run_gemma_centered_sequential_review(question: str, initial_raw: dict) -> tuple[str, str, float | None]:
    total_time = 0.0

    gemma_initial = initial_raw.get(PRIMARY_MODEL, "")
    llama_initial = initial_raw.get("llama3.1:8b", "")
    mistral_initial = initial_raw.get("mistral-nemo:12b", "")

    try:
        start = time.time()
        llama_revised = ask_ollama(
            "llama3.1:8b",
            build_sequential_llama_revision_prompt(question, llama_initial, gemma_initial),
        )
        total_time += round(time.time() - start, 2)
    except Exception as e:
        llama_revised = f"ERROR: {e}"

    try:
        start = time.time()
        mistral_revised = ask_ollama(
            "mistral-nemo:12b",
            build_sequential_mistral_revision_prompt(
                question=question,
                mistral_initial=mistral_initial,
                gemma_initial=gemma_initial,
                llama_initial=llama_initial,
                llama_revised=llama_revised,
            ),
        )
        total_time += round(time.time() - start, 2)
    except Exception as e:
        mistral_revised = f"ERROR: {e}"

    try:
        start = time.time()
        gemma_final = ask_ollama(
            PRIMARY_MODEL,
            build_sequential_gemma_final_prompt(
                question=question,
                gemma_initial=gemma_initial,
                llama_initial=llama_initial,
                llama_revised=llama_revised,
                mistral_initial=mistral_initial,
                mistral_revised=mistral_revised,
            ),
        )
        total_time += round(time.time() - start, 2)
    except Exception as e:
        gemma_final = f"ERROR: {e}"

    answer = extract_choice(gemma_final)

    raw_summary = f"""
--- Gemma initial ---
{gemma_initial}

--- Llama initial ---
{llama_initial}

--- Llama revised after seeing Gemma ---
{llama_revised}

--- Mistral initial ---
{mistral_initial}

--- Mistral revised after seeing Gemma + Llama ---
{mistral_revised}

--- Gemma final decision ---
{gemma_final}
"""

    return answer, raw_summary, round(total_time, 2)


def detect_base_pattern(model_choices: dict) -> str:
    g = model_choices.get("gemma3:12b", "?")
    l = model_choices.get("llama3.1:8b", "?")
    m = model_choices.get("mistral-nemo:12b", "?")

    if g == l == m and g in VALID_CHOICES:
        return "all_same"

    if g == l and g != m:
        return "gemma_llama_agree"

    if g == m and g != l:
        return "gemma_mistral_agree"

    if l == m and g != l:
        return "gemma_minority_vs_lm"

    return "all_different"


def coarse_pattern_best_router(
    pattern: str,
    judge_always_v2_answer: str,
    gemma_final_arbiter_v2_answer: str,
    sequential_answer: str,
) -> tuple[str, str]:
    """
    Coarse Pattern-Best Router.
    This router does NOT see the correct answer at runtime.
    It selects a system based only on the base model agreement pattern.

    Mapping based on previous post-hoc 40-question analysis:
    - all_same -> Gemma-Centered Sequential Review
    - gemma_llama_agree -> Gemma Final Arbiter v2
    - gemma_minority_vs_lm -> Judge Always v2
    - gemma_mistral_agree -> Judge Always v2
    - all_different -> Judge Always v2
    """

    if pattern == "all_same":
        return sequential_answer, "all_same -> Gemma-Centered Sequential Review"

    if pattern == "gemma_llama_agree":
        return gemma_final_arbiter_v2_answer, "gemma_llama_agree -> Gemma Final Arbiter v2"

    if pattern == "gemma_minority_vs_lm":
        return judge_always_v2_answer, "gemma_minority_vs_lm -> Judge Always v2"

    if pattern == "gemma_mistral_agree":
        return judge_always_v2_answer, "gemma_mistral_agree -> Judge Always v2"

    return judge_always_v2_answer, "all_different -> Judge Always v2"


def load_questions(path: str = "questions.jsonl") -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]

    return questions


def main():
    questions = load_questions()
    rows = []

    print(f"\nTotal questions: {len(questions)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Primary model: {PRIMARY_MODEL}")
    print(f"Run name: {RUN_NAME}")
    print("Systems measured:")
    print("- Judge Always v2")
    print("- Gemma Final Arbiter v2")
    print("- Coarse Pattern-Best Router")
    print("\nNote: The router uses Gemma-Centered Sequential Review internally only for all_same pattern.\n")

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", "unknown")
        question = item["question"]
        correct_answer = item["answer"].upper()

        print("=" * 100)
        print(f"Question {index}/{len(questions)} | ID: {question_id} | {category} | {difficulty}")
        print(f"Correct answer: {correct_answer}")

        model_raw = {}
        model_choices = {}
        model_confidences = {}
        model_times = {}

        # Base model answers, needed for both v2 prompts and router pattern detection.
        for model in MODELS:
            try:
                start = time.time()
                raw = ask_ollama(model, build_initial_prompt(question))
                elapsed = round(time.time() - start, 2)

                choice = extract_choice(raw)
                confidence = extract_confidence(raw)

                model_raw[model] = raw
                model_choices[model] = choice
                model_confidences[model] = confidence
                model_times[model] = elapsed

                print(f"{model}: {choice} | confidence {confidence} | {'✅' if choice == correct_answer else '❌'} ({elapsed} sec)")
            except Exception as e:
                model_raw[model] = f"ERROR: {e}"
                model_choices[model] = "?"
                model_confidences[model] = 50
                model_times[model] = None
                print(f"{model}: ERROR {e}")

        # Judge Always v2
        try:
            start = time.time()
            judge_always_v2_raw = ask_ollama(
                PRIMARY_MODEL,
                build_judge_always_prompt_v2(question, model_raw, model_choices, model_confidences),
            )
            judge_always_v2_time = round(time.time() - start, 2)
            judge_always_v2_answer = extract_choice(judge_always_v2_raw)
            judge_always_v2_confidence = extract_confidence(judge_always_v2_raw)
        except Exception as e:
            judge_always_v2_raw = f"ERROR: {e}"
            judge_always_v2_time = None
            judge_always_v2_answer = "?"
            judge_always_v2_confidence = 50

        # Gemma Final Arbiter v2
        try:
            start = time.time()
            gemma_final_v2_raw = ask_ollama(
                PRIMARY_MODEL,
                build_gemma_final_arbiter_prompt_v2(question, model_raw, model_choices, model_confidences),
            )
            gemma_final_v2_time = round(time.time() - start, 2)
            gemma_final_v2_answer = extract_choice(gemma_final_v2_raw)
            gemma_final_v2_confidence = extract_confidence(gemma_final_v2_raw)
        except Exception as e:
            gemma_final_v2_raw = f"ERROR: {e}"
            gemma_final_v2_time = None
            gemma_final_v2_answer = "?"
            gemma_final_v2_confidence = 50

        # Router pattern.
        pattern = detect_base_pattern(model_choices)

        # The Coarse Pattern-Best Router uses sequential review internally only for all_same.
        # For other patterns, it only reuses Judge Always v2 or Gemma Final Arbiter v2.
        if pattern == "all_same":
            sequential_answer, sequential_raw, sequential_time = run_gemma_centered_sequential_review(question, model_raw)
        else:
            sequential_answer = "?"
            sequential_raw = "Not called. Coarse Pattern-Best Router only calls sequential review for all_same pattern."
            sequential_time = 0

        router_answer, router_reason = coarse_pattern_best_router(
            pattern=pattern,
            judge_always_v2_answer=judge_always_v2_answer,
            gemma_final_arbiter_v2_answer=gemma_final_v2_answer,
            sequential_answer=sequential_answer,
        )

        print(f"Judge Always v2: {judge_always_v2_answer} {'✅' if judge_always_v2_answer == correct_answer else '❌'}")
        print(f"Gemma Final Arbiter v2: {gemma_final_v2_answer} {'✅' if gemma_final_v2_answer == correct_answer else '❌'}")
        print(f"Coarse Pattern-Best Router: {router_answer} {'✅' if router_answer == correct_answer else '❌'} | {router_reason}")

        row = {
            "id": question_id,
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "correct_answer": correct_answer,

            "base_pattern": pattern,

            "judge_always_v2_answer": judge_always_v2_answer,
            "judge_always_v2_correct": judge_always_v2_answer == correct_answer,
            "judge_always_v2_confidence": judge_always_v2_confidence,
            "judge_always_v2_time_sec": judge_always_v2_time,
            "judge_always_v2_raw": judge_always_v2_raw,

            "gemma_final_arbiter_v2_answer": gemma_final_v2_answer,
            "gemma_final_arbiter_v2_correct": gemma_final_v2_answer == correct_answer,
            "gemma_final_arbiter_v2_confidence": gemma_final_v2_confidence,
            "gemma_final_arbiter_v2_time_sec": gemma_final_v2_time,
            "gemma_final_arbiter_v2_raw": gemma_final_v2_raw,

            "coarse_pattern_best_router_answer": router_answer,
            "coarse_pattern_best_router_correct": router_answer == correct_answer,
            "coarse_pattern_best_router_reason": router_reason,

            "internal_sequential_answer": sequential_answer,
            "internal_sequential_time_sec": sequential_time,
            "internal_sequential_raw": sequential_raw,
        }

        for model in MODELS:
            row[f"{model}_answer"] = model_choices[model]
            row[f"{model}_confidence"] = model_confidences[model]
            row[f"{model}_correct"] = model_choices[model] == correct_answer
            row[f"{model}_time_sec"] = model_times[model]
            row[f"{model}_raw"] = model_raw[model]

        rows.append(row)

    df = pd.DataFrame(rows)

    results_path = f"benchmark_results_{RUN_NAME}.csv"
    summary_path = f"benchmark_summary_{RUN_NAME}.csv"
    category_path = f"benchmark_by_category_{RUN_NAME}.csv"
    difficulty_path = f"benchmark_by_difficulty_{RUN_NAME}.csv"
    pattern_path = f"benchmark_by_pattern_{RUN_NAME}.csv"

    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    systems = [
        {
            "system": "Judge Always v2",
            "correct_col": "judge_always_v2_correct",
            "time_col": "judge_always_v2_time_sec",
        },
        {
            "system": "Gemma Final Arbiter v2",
            "correct_col": "gemma_final_arbiter_v2_correct",
            "time_col": "gemma_final_arbiter_v2_time_sec",
        },
        {
            "system": "Coarse Pattern-Best Router",
            "correct_col": "coarse_pattern_best_router_correct",
            "time_col": None,
        },
    ]

    summary_rows = []
    for system in systems:
        correct_col = system["correct_col"]
        time_col = system["time_col"]

        accuracy = df[correct_col].mean()

        if time_col and time_col in df.columns:
            avg_time = df[time_col].dropna().mean()
            avg_time = round(avg_time, 2) if pd.notna(avg_time) else None
        else:
            avg_time = 0

        summary_rows.append({
            "system": system["system"],
            "accuracy": round(accuracy * 100, 2),
            "correct_count": int(df[correct_col].sum()),
            "total_questions": len(df),
            "avg_extra_time_sec": avg_time,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="accuracy", ascending=False)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    category_rows = []
    for category_name, group in df.groupby("category"):
        for system in systems:
            category_rows.append({
                "category": category_name,
                "system": system["system"],
                "accuracy": round(group[system["correct_col"]].mean() * 100, 2),
                "correct_count": int(group[system["correct_col"]].sum()),
                "question_count": len(group),
            })

    pd.DataFrame(category_rows).to_csv(category_path, index=False, encoding="utf-8-sig")

    difficulty_rows = []
    for difficulty_name, group in df.groupby("difficulty"):
        for system in systems:
            difficulty_rows.append({
                "difficulty": difficulty_name,
                "system": system["system"],
                "accuracy": round(group[system["correct_col"]].mean() * 100, 2),
                "correct_count": int(group[system["correct_col"]].sum()),
                "question_count": len(group),
            })

    pd.DataFrame(difficulty_rows).to_csv(difficulty_path, index=False, encoding="utf-8-sig")

    pattern_rows = []
    for pattern_name, group in df.groupby("base_pattern"):
        for system in systems:
            pattern_rows.append({
                "base_pattern": pattern_name,
                "system": system["system"],
                "accuracy": round(group[system["correct_col"]].mean() * 100, 2),
                "correct_count": int(group[system["correct_col"]].sum()),
                "question_count": len(group),
            })

    pd.DataFrame(pattern_rows).to_csv(pattern_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("=== SUMMARY ===")
    print(summary_df)

    print("\nFiles created:")
    print(f"- {results_path}")
    print(f"- {summary_path}")
    print(f"- {category_path}")
    print(f"- {difficulty_path}")
    print(f"- {pattern_path}")


if __name__ == "__main__":
    main()
