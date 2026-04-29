import json
import re
import time
from collections import Counter

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "mistral-nemo:12b",
]

JUDGE_MODEL = "qwen3:8b"

# İlk test için 10 bırak. Her şey çalışırsa None yapıp 100 soruyu çalıştır.
MAX_QUESTIONS = 10

# Çok yavaş olursa bazılarını False yapabilirsin.
RUN_JUDGE_ALWAYS = True
RUN_REASONING_JUDGE = True
RUN_PEER_REVIEW_REVISION = True
RUN_DEBATE_REVISION = True
RUN_CONFIDENCE_WEIGHTED_VOTE = True

VALID_CHOICES = ["A", "B", "C", "D"]


def ask_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=900)
    response.raise_for_status()
    return response.json()["response"]


def extract_choice(text: str) -> str:
    if not text:
        return "?"

    text_upper = text.strip().upper()

    patterns = [
        r"CEVAP\s*[:\-]?\s*([ABCD])",
        r"FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])",
        r"ANSWER\s*[:\-]?\s*([ABCD])",
        r"SEÇENEK\s*[:\-]?\s*([ABCD])",
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
        r"G[ÜU]VEN\s*[:\-]?\s*(\d{1,3})",
        r"CONFIDENCE\s*[:\-]?\s*(\d{1,3})",
        r"G[ÜU]VEN\s*PUANI\s*[:\-]?\s*(\d{1,3})",
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
Aşağıdaki çoktan seçmeli soruyu çöz.

Kurallar:
- Sadece A, B, C veya D seçeneklerinden birini seç.
- Kısa bir gerekçe yaz.
- En sonda MUTLAKA şu formatı kullan:

GEREKCE: kısa gerekçen
CEVAP: X
GUVEN: 0-100

Soru:
{question}
"""


def build_simple_judge_prompt(question: str, choices: dict, confidences: dict | None = None) -> str:
    lines = []
    for model in MODELS:
        conf = ""
        if confidences is not None:
            conf = f", güven: {confidences.get(model, 50)}"
        lines.append(f"{model}: {choices.get(model, '?')}{conf}")

    joined = "\n".join(lines)

    return f"""
Sen çoktan seçmeli sorular için hakem modelsin.

Soru:
{question}

Modellerin seçtiği şıklar:
{joined}

Görevin:
- Çoğunluğa körü körüne uyma.
- Soruyu kendin kontrol et.
- Sadece A, B, C veya D seçeneklerinden birini seç.
- En sonda MUTLAKA şu formatta cevap ver:
CEVAP: X
"""


def build_reasoning_judge_prompt(question: str, model_raw_answers: dict, model_choices: dict, model_confidences: dict) -> str:
    blocks = []

    for model in MODELS:
        blocks.append(
            f"""
--- {model} ---
Seçtiği şık: {model_choices.get(model, "?")}
Güven: {model_confidences.get(model, 50)}
Cevap ve gerekçe:
{model_raw_answers.get(model, "")}
"""
        )

    joined = "\n".join(blocks)

    return f"""
Sen çoktan seçmeli sorularda gerekçe kalitesini değerlendiren dikkatli bir hakem modelsin.

Soru:
{question}

Üç modelin cevapları, gerekçeleri ve güven puanları:
{joined}

Görevin:
- Sadece şıklara değil, gerekçelerin doğruluğuna da bak.
- İki model aynı cevabı verdi diye otomatik doğru kabul etme.
- Azınlıkta kalan model daha doğru gerekçelendirdiyse onu seçebilirsin.
- En doğru final cevabı A, B, C veya D olarak ver.
- En sonda MUTLAKA şu formatı kullan:

CEVAP: X
"""


def build_revision_prompt(model_name: str, question: str, own_raw: str, own_choice: str, other_raw_answers: dict, mode: str) -> str:
    other_blocks = []
    for model, raw in other_raw_answers.items():
        if model == model_name:
            continue
        other_blocks.append(
            f"""
--- {model} ---
Cevabı:
{raw}
"""
        )

    others = "\n".join(other_blocks)

    if mode == "peer":
        instruction = """
Sen kendi ilk cevabını savunmak zorunda değilsin.
Diğer modellerin cevaplarını birer değerlendirme/review gibi kullan.
Eğer ilk cevabın yanlış görünüyorsa değiştir.
Eğer ilk cevabın doğru görünüyorsa aynı bırak.
"""
    else:
        instruction = """
Bu bir model tartışması/debate turudur.
Üç modelin bağımsız cevaplarını görüyorsun.
Hepsini karşılaştır ve en doğru final cevabı seç.
Çoğunluğa körü körüne uyma.
"""

    return f"""
Soru:
{question}

Senin ilk cevabın:
{own_raw}

Diğer modellerin cevapları:
{others}

{instruction}

Çıktı formatı:
GEREKCE: kısa gerekçen
CEVAP: X
GUVEN: 0-100
"""


def majority_vote(choices: list[str]) -> str:
    valid_choices = [choice for choice in choices if choice in VALID_CHOICES]

    if not valid_choices:
        return "?"

    counts = Counter(valid_choices)
    most_common = counts.most_common()

    if len(most_common) == 1:
        return most_common[0][0]

    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    return "TIE"


def all_models_agree(choices: list[str]) -> bool:
    valid_choices = [choice for choice in choices if choice in VALID_CHOICES]

    if len(valid_choices) != len(MODELS):
        return False

    return len(set(valid_choices)) == 1


def confidence_weighted_vote(choices: dict, confidences: dict) -> str:
    scores = {choice: 0 for choice in VALID_CHOICES}

    for model, choice in choices.items():
        if choice in VALID_CHOICES:
            scores[choice] += confidences.get(model, 50)

    best_choice = max(scores, key=scores.get)
    best_score = scores[best_choice]

    if best_score == 0:
        return "?"

    # Eşitlik kontrolü
    tied = [choice for choice, score in scores.items() if score == best_score]
    if len(tied) > 1:
        return "TIE"

    return best_choice


def resolve_revised_choices(question: str, revised_raw: dict, revised_choices: dict, revised_confidences: dict) -> tuple[str, str, float | None]:
    majority_answer = majority_vote(list(revised_choices.values()))

    if majority_answer != "TIE":
        return majority_answer, "Revize cevaplarda çoğunluk bulundu.", 0

    prompt = build_reasoning_judge_prompt(question, revised_raw, revised_choices, revised_confidences)

    try:
        start = time.time()
        raw = ask_ollama(JUDGE_MODEL, prompt)
        elapsed = round(time.time() - start, 2)
        answer = extract_choice(raw)
        return answer, raw, elapsed
    except Exception as e:
        return "?", f"HATA: {e}", None


def load_questions(path: str = "questions.jsonl") -> list[dict]:
    questions = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]

    return questions


def run_revision_round(mode: str, question: str, initial_raw: dict, initial_choices: dict) -> tuple[dict, dict, dict, dict]:
    revised_raw = {}
    revised_choices = {}
    revised_confidences = {}
    revised_times = {}

    for model in MODELS:
        prompt = build_revision_prompt(
            model_name=model,
            question=question,
            own_raw=initial_raw.get(model, ""),
            own_choice=initial_choices.get(model, "?"),
            other_raw_answers=initial_raw,
            mode=mode,
        )

        try:
            start = time.time()
            raw = ask_ollama(model, prompt)
            elapsed = round(time.time() - start, 2)

            revised_raw[model] = raw
            revised_choices[model] = extract_choice(raw)
            revised_confidences[model] = extract_confidence(raw)
            revised_times[model] = elapsed

        except Exception as e:
            revised_raw[model] = f"HATA: {e}"
            revised_choices[model] = "?"
            revised_confidences[model] = 50
            revised_times[model] = None

    return revised_raw, revised_choices, revised_confidences, revised_times


def main():
    questions = load_questions()
    rows = []

    print(f"\nToplam soru sayısı: {len(questions)}")
    print(f"Modeller: {', '.join(MODELS)}")
    print(f"Hakem model: {JUDGE_MODEL}")
    print("Ölçülen sistemler:")
    print("- Tekil modeller")
    print("- Majority Vote")
    print("- Judge on Disagreement")
    print("- Judge Always")
    print("- Reasoning Judge")
    print("- Peer Review Revision")
    print("- Debate Revision")
    print("- Confidence Weighted Vote\n")

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", "unknown")
        question = item["question"]
        correct_answer = item["answer"].upper()

        print("=" * 90)
        print(f"Soru {index}/{len(questions)} | ID: {question_id} | {category} | {difficulty}")
        print(f"Doğru cevap: {correct_answer}")

        model_raw_answers = {}
        model_choices = {}
        model_confidences = {}
        model_times = {}

        # 1. Tekil model cevapları
        for model in MODELS:
            prompt = build_initial_prompt(question)

            try:
                start = time.time()
                raw_answer = ask_ollama(model, prompt)
                elapsed = round(time.time() - start, 2)

                choice = extract_choice(raw_answer)
                confidence = extract_confidence(raw_answer)

                model_raw_answers[model] = raw_answer
                model_choices[model] = choice
                model_confidences[model] = confidence
                model_times[model] = elapsed

                correctness = "✅" if choice == correct_answer else "❌"
                print(f"{model}: {choice} | güven {confidence} | {correctness} ({elapsed} sn)")

            except Exception as e:
                model_raw_answers[model] = f"HATA: {e}"
                model_choices[model] = "?"
                model_confidences[model] = 50
                model_times[model] = None
                print(f"{model}: HATA {e}")

        choices = list(model_choices.values())

        # 2. Majority Vote
        majority_answer = majority_vote(choices)
        if majority_answer == "TIE":
            majority_answer = "?"

        # 3. Confidence Weighted Vote
        if RUN_CONFIDENCE_WEIGHTED_VOTE:
            confidence_weighted_answer = confidence_weighted_vote(model_choices, model_confidences)
            if confidence_weighted_answer == "TIE":
                confidence_weighted_answer = "?"
        else:
            confidence_weighted_answer = "?"

        # 4. Judge on Disagreement
        if all_models_agree(choices):
            judge_on_disagreement_answer = choices[0]
            judge_on_disagreement_raw = "Hakem çağrılmadı. Üç model aynı fikirdeydi."
            judge_on_disagreement_time = 0
        else:
            prompt = build_simple_judge_prompt(question, model_choices, model_confidences)
            try:
                start = time.time()
                judge_on_disagreement_raw = ask_ollama(JUDGE_MODEL, prompt)
                judge_on_disagreement_time = round(time.time() - start, 2)
                judge_on_disagreement_answer = extract_choice(judge_on_disagreement_raw)
            except Exception as e:
                judge_on_disagreement_raw = f"HATA: {e}"
                judge_on_disagreement_time = None
                judge_on_disagreement_answer = "?"

        # 5. Judge Always
        if RUN_JUDGE_ALWAYS:
            prompt = build_simple_judge_prompt(question, model_choices, model_confidences)
            try:
                start = time.time()
                judge_always_raw = ask_ollama(JUDGE_MODEL, prompt)
                judge_always_time = round(time.time() - start, 2)
                judge_always_answer = extract_choice(judge_always_raw)
            except Exception as e:
                judge_always_raw = f"HATA: {e}"
                judge_always_time = None
                judge_always_answer = "?"
        else:
            judge_always_raw = "RUN_JUDGE_ALWAYS kapalı."
            judge_always_time = None
            judge_always_answer = "?"

        # 6. Reasoning Judge
        if RUN_REASONING_JUDGE:
            prompt = build_reasoning_judge_prompt(question, model_raw_answers, model_choices, model_confidences)
            try:
                start = time.time()
                reasoning_judge_raw = ask_ollama(JUDGE_MODEL, prompt)
                reasoning_judge_time = round(time.time() - start, 2)
                reasoning_judge_answer = extract_choice(reasoning_judge_raw)
            except Exception as e:
                reasoning_judge_raw = f"HATA: {e}"
                reasoning_judge_time = None
                reasoning_judge_answer = "?"
        else:
            reasoning_judge_raw = "RUN_REASONING_JUDGE kapalı."
            reasoning_judge_time = None
            reasoning_judge_answer = "?"

        # 7. Peer Review Revision
        if RUN_PEER_REVIEW_REVISION:
            peer_raw, peer_choices, peer_confidences, peer_times = run_revision_round(
                mode="peer",
                question=question,
                initial_raw=model_raw_answers,
                initial_choices=model_choices,
            )
            peer_review_answer, peer_review_resolver_raw, peer_review_resolver_time = resolve_revised_choices(
                question=question,
                revised_raw=peer_raw,
                revised_choices=peer_choices,
                revised_confidences=peer_confidences,
            )
            peer_review_time = sum(t for t in peer_times.values() if t is not None) + (peer_review_resolver_time or 0)
        else:
            peer_raw, peer_choices, peer_confidences, peer_times = {}, {}, {}, {}
            peer_review_answer = "?"
            peer_review_resolver_raw = "RUN_PEER_REVIEW_REVISION kapalı."
            peer_review_time = None

        # 8. Debate Revision
        if RUN_DEBATE_REVISION:
            debate_raw, debate_choices, debate_confidences, debate_times = run_revision_round(
                mode="debate",
                question=question,
                initial_raw=model_raw_answers,
                initial_choices=model_choices,
            )
            debate_answer, debate_resolver_raw, debate_resolver_time = resolve_revised_choices(
                question=question,
                revised_raw=debate_raw,
                revised_choices=debate_choices,
                revised_confidences=debate_confidences,
            )
            debate_time = sum(t for t in debate_times.values() if t is not None) + (debate_resolver_time or 0)
        else:
            debate_raw, debate_choices, debate_confidences, debate_times = {}, {}, {}, {}
            debate_answer = "?"
            debate_resolver_raw = "RUN_DEBATE_REVISION kapalı."
            debate_time = None

        print(f"Majority Vote: {majority_answer} {'✅' if majority_answer == correct_answer else '❌'}")
        print(f"Confidence Weighted Vote: {confidence_weighted_answer} {'✅' if confidence_weighted_answer == correct_answer else '❌'}")
        print(f"Judge on Disagreement: {judge_on_disagreement_answer} {'✅' if judge_on_disagreement_answer == correct_answer else '❌'}")
        print(f"Judge Always: {judge_always_answer} {'✅' if judge_always_answer == correct_answer else '❌'}")
        print(f"Reasoning Judge: {reasoning_judge_answer} {'✅' if reasoning_judge_answer == correct_answer else '❌'}")
        print(f"Peer Review Revision: {peer_review_answer} {'✅' if peer_review_answer == correct_answer else '❌'}")
        print(f"Debate Revision: {debate_answer} {'✅' if debate_answer == correct_answer else '❌'}")

        row = {
            "id": question_id,
            "category": category,
            "difficulty": difficulty,
            "question": question,
            "correct_answer": correct_answer,

            "majority_vote_answer": majority_answer,
            "majority_vote_correct": majority_answer == correct_answer,

            "confidence_weighted_vote_answer": confidence_weighted_answer,
            "confidence_weighted_vote_correct": confidence_weighted_answer == correct_answer,

            "judge_on_disagreement_answer": judge_on_disagreement_answer,
            "judge_on_disagreement_correct": judge_on_disagreement_answer == correct_answer,
            "judge_on_disagreement_time_sec": judge_on_disagreement_time,
            "judge_on_disagreement_raw": judge_on_disagreement_raw,

            "judge_always_answer": judge_always_answer,
            "judge_always_correct": judge_always_answer == correct_answer,
            "judge_always_time_sec": judge_always_time,
            "judge_always_raw": judge_always_raw,

            "reasoning_judge_answer": reasoning_judge_answer,
            "reasoning_judge_correct": reasoning_judge_answer == correct_answer,
            "reasoning_judge_time_sec": reasoning_judge_time,
            "reasoning_judge_raw": reasoning_judge_raw,

            "peer_review_revision_answer": peer_review_answer,
            "peer_review_revision_correct": peer_review_answer == correct_answer,
            "peer_review_revision_time_sec": peer_review_time,
            "peer_review_resolver_raw": peer_review_resolver_raw,

            "debate_revision_answer": debate_answer,
            "debate_revision_correct": debate_answer == correct_answer,
            "debate_revision_time_sec": debate_time,
            "debate_resolver_raw": debate_resolver_raw,
        }

        for model in MODELS:
            row[f"{model}_answer"] = model_choices[model]
            row[f"{model}_confidence"] = model_confidences[model]
            row[f"{model}_correct"] = model_choices[model] == correct_answer
            row[f"{model}_time_sec"] = model_times[model]
            row[f"{model}_raw"] = model_raw_answers[model]

            row[f"peer_{model}_answer"] = peer_choices.get(model, "?")
            row[f"peer_{model}_confidence"] = peer_confidences.get(model, 50)
            row[f"peer_{model}_raw"] = peer_raw.get(model, "")

            row[f"debate_{model}_answer"] = debate_choices.get(model, "?")
            row[f"debate_{model}_confidence"] = debate_confidences.get(model, 50)
            row[f"debate_{model}_raw"] = debate_raw.get(model, "")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")

    systems = []

    for model in MODELS:
        systems.append({
            "system": model,
            "correct_col": f"{model}_correct",
            "time_col": f"{model}_time_sec",
        })

    systems.extend([
        {"system": "Majority Vote", "correct_col": "majority_vote_correct", "time_col": None},
        {"system": "Judge on Disagreement", "correct_col": "judge_on_disagreement_correct", "time_col": "judge_on_disagreement_time_sec"},
        {"system": "Judge Always", "correct_col": "judge_always_correct", "time_col": "judge_always_time_sec"},
        {"system": "Reasoning Judge", "correct_col": "reasoning_judge_correct", "time_col": "reasoning_judge_time_sec"},
        {"system": "Peer Review Revision", "correct_col": "peer_review_revision_correct", "time_col": "peer_review_revision_time_sec"},
        {"system": "Debate Revision", "correct_col": "debate_revision_correct", "time_col": "debate_revision_time_sec"},
        {"system": "Confidence Weighted Vote", "correct_col": "confidence_weighted_vote_correct", "time_col": None},
    ])

    summary = []
    for system in systems:
        correct_col = system["correct_col"]
        time_col = system["time_col"]

        accuracy = df[correct_col].mean()

        if time_col and time_col in df.columns:
            avg_time = df[time_col].dropna().mean()
            avg_time = round(avg_time, 2) if pd.notna(avg_time) else None
        else:
            avg_time = 0

        summary.append({
            "system": system["system"],
            "accuracy": round(accuracy * 100, 2),
            "avg_extra_time_sec": avg_time,
            "total_questions": len(df),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("benchmark_summary.csv", index=False, encoding="utf-8-sig")

    category_rows = []
    for category_name, group in df.groupby("category"):
        for system in systems:
            correct_col = system["correct_col"]
            category_rows.append({
                "category": category_name,
                "system": system["system"],
                "accuracy": round(group[correct_col].mean() * 100, 2),
                "question_count": len(group),
            })

    category_df = pd.DataFrame(category_rows)
    category_df.to_csv("benchmark_by_category.csv", index=False, encoding="utf-8-sig")

    difficulty_rows = []
    for difficulty_name, group in df.groupby("difficulty"):
        for system in systems:
            correct_col = system["correct_col"]
            difficulty_rows.append({
                "difficulty": difficulty_name,
                "system": system["system"],
                "accuracy": round(group[correct_col].mean() * 100, 2),
                "question_count": len(group),
            })

    difficulty_df = pd.DataFrame(difficulty_rows)
    difficulty_df.to_csv("benchmark_by_difficulty.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("=== GENEL ÖZET ===")
    print(summary_df)

    print("\nDosyalar oluşturuldu:")
    print("- benchmark_results.csv")
    print("- benchmark_summary.csv")
    print("- benchmark_by_category.csv")
    print("- benchmark_by_difficulty.csv")


if __name__ == "__main__":
    main()
