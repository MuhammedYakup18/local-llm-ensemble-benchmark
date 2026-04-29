import json
import re
import time
from collections import Counter

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
RUN_NAME = "gemma_mmlu_full40"
PRIMARY_MODEL = "gemma3:12b"
MODELS = ["gemma3:12b", "llama3.1:8b", "mistral-nemo:12b"]
JUDGE_MODEL = PRIMARY_MODEL
MAX_QUESTIONS = 40
VALID_CHOICES = ["A", "B", "C", "D"]


def ask_ollama(model: str, prompt: str, temperature: float = 0, timeout: int = 900) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 1},
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
    return matches[-1] if matches else "?"


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
            return max(0, min(100, int(matches[-1])))
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


def build_simple_judge_prompt(question: str, choices: dict, confidences: dict | None = None) -> str:
    lines = []
    for model in MODELS:
        conf = f", confidence: {confidences.get(model, 50)}" if confidences is not None else ""
        lines.append(f"{model}: {choices.get(model, '?')}{conf}")
    return f"""
You are a careful judge for a multiple-choice benchmark question.

Question:
{question}

Model choices:
{chr(10).join(lines)}

Task:
- Do not blindly follow the majority.
- Check the question yourself.
- Choose exactly one option: A, B, C, or D.
- End your response exactly as:
ANSWER: X
"""


def build_reasoning_judge_prompt(question: str, model_raw_answers: dict, model_choices: dict, model_confidences: dict) -> str:
    blocks = []
    for model in MODELS:
        blocks.append(f"""
--- {model} ---
Choice: {model_choices.get(model, "?")}
Confidence: {model_confidences.get(model, 50)}
Answer and reason:
{model_raw_answers.get(model, "")}
""")
    return f"""
You are a careful evaluator of reasoning quality for multiple-choice benchmark questions.

Question:
{question}

Three models answered independently:
{''.join(blocks)}

Task:
- Evaluate the reasoning, not only the final letters.
- Do not blindly follow the majority.
- If a minority answer has stronger reasoning, choose it.
- Choose exactly one option: A, B, C, or D.
- End your response exactly as:
ANSWER: X
"""


def build_revision_prompt(model_name: str, question: str, own_raw: str, other_raw_answers: dict, mode: str) -> str:
    other_blocks = []
    for model, raw in other_raw_answers.items():
        if model != model_name:
            other_blocks.append(f"""
--- {model} ---
Answer:
{raw}
""")
    instruction = """
This is a peer-review revision round.
You do not have to defend your original answer.
Use the other models' answers as reviews.
If your first answer appears wrong, revise it.
If your first answer still appears correct, keep it.
Explain whether you changed your answer or kept it.
""" if mode == "peer" else """
This is a debate/revision round.
You can see the independent answers from the other models.
Compare all answers and choose the most likely correct option.
Do not blindly follow the majority.
"""
    return f"""
Question:
{question}

Your first answer:
{own_raw}

Other models' answers:
{''.join(other_blocks)}

{instruction}

Output format:
REASON: your brief reason
ANSWER: X
CONFIDENCE: 0-100
"""


def build_gemma_final_arbiter_prompt(question: str, initial_raw: dict, initial_choices: dict, initial_confidences: dict) -> str:
    blocks = []
    for model in MODELS:
        blocks.append(f"""
--- {model} ---
Initial choice: {initial_choices.get(model, "?")}
Confidence: {initial_confidences.get(model, 50)}
Answer and reason:
{initial_raw.get(model, "")}
""")
    return f"""
You are {PRIMARY_MODEL}, the primary final arbiter.

Question:
{question}

All initial model answers:
{''.join(blocks)}

Your task:
- You already have your own initial answer above.
- You are not required to defend your initial answer.
- If another model gives stronger reasoning, you may change your answer.
- Do not blindly follow majority vote.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your final reason
ANSWER: X
CONFIDENCE: 0-100
"""


def build_gemma_error_check_prompt(question: str, initial_raw: dict, initial_choices: dict, initial_confidences: dict) -> str:
    maj = majority_vote(list(initial_choices.values()))
    maj = maj if maj != "TIE" else "no clear majority"
    return f"""
You are {PRIMARY_MODEL}, the primary model.

Question:
{question}

Your initial answer:
{initial_raw.get(PRIMARY_MODEL, "")}

Other models:
--- llama3.1:8b ---
Choice: {initial_choices.get("llama3.1:8b", "?")}
Confidence: {initial_confidences.get("llama3.1:8b", 50)}
Answer:
{initial_raw.get("llama3.1:8b", "")}

--- mistral-nemo:12b ---
Choice: {initial_choices.get("mistral-nemo:12b", "?")}
Confidence: {initial_confidences.get("mistral-nemo:12b", 50)}
Answer:
{initial_raw.get("mistral-nemo:12b", "")}

Initial majority: {maj}

Task:
- Check whether your own initial answer may contain an error.
- Do not change just because others disagree.
- Do change if the other models reveal a stronger argument or a mistake in your reasoning.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your final error-check reason
ANSWER: X
CONFIDENCE: 0-100
"""


def build_gemma_final_after_revisions_prompt(question: str, initial_raw: dict, revised_raw: dict, revised_choices: dict, system_name: str) -> str:
    blocks = []
    for model in MODELS:
        blocks.append(f"""
--- {model} ---
Initial answer:
{initial_raw.get(model, "")}

Revised answer in {system_name}:
Choice: {revised_choices.get(model, "?")}
{revised_raw.get(model, "")}
""")
    return f"""
You are {PRIMARY_MODEL}, the final arbiter.

Question:
{question}

Initial and revised answers:
{''.join(blocks)}

Task:
- Review the initial answers and the revised answers.
- Do not blindly follow the revised majority.
- If your own initial answer is wrong, change it.
- If another model's revised reasoning is stronger, use it.
- Choose exactly one option: A, B, C, or D.
- End exactly as:
REASON: your final reason
ANSWER: X
CONFIDENCE: 0-100
"""


def build_option_wise_prompt(question: str) -> str:
    return f"""
You are verifying a multiple-choice question option by option.

Question:
{question}

Task:
- Evaluate each option A, B, C, and D separately.
- Give each option a correctness score from 0 to 100.
- Then choose the single best answer.
- Use this exact output format:
A_SCORE: number
B_SCORE: number
C_SCORE: number
D_SCORE: number
REASON: brief reason
ANSWER: X
CONFIDENCE: 0-100
"""


def parse_option_scores(text: str, chosen_answer: str = "?", confidence: int = 50) -> dict:
    scores = {choice: 0 for choice in VALID_CHOICES}
    if text:
        upper = text.upper()
        for choice in VALID_CHOICES:
            for pattern in [rf"{choice}_SCORE\s*[:\-]?\s*(\d{{1,3}})", rf"{choice}\s*SCORE\s*[:\-]?\s*(\d{{1,3}})", rf"{choice}\)\s*[:\-]?\s*(\d{{1,3}})"]:
                matches = re.findall(pattern, upper)
                if matches:
                    scores[choice] = max(0, min(100, int(matches[-1])))
                    break
    if sum(scores.values()) == 0 and chosen_answer in VALID_CHOICES:
        scores[chosen_answer] = confidence
    return scores


def build_self_consistency_prompt(question: str, sample_id: int) -> str:
    return f"""
Solve the following multiple-choice question. This is independent reasoning sample #{sample_id}.

Rules:
- Choose exactly one option: A, B, C, or D.
- Think briefly and independently.
- End exactly as:
ANSWER: X

Question:
{question}
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


def build_sequential_mistral_revision_prompt(question: str, mistral_initial: str, gemma_initial: str, llama_initial: str, llama_revised: str) -> str:
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


def build_sequential_gemma_final_prompt(question: str, gemma_initial: str, llama_initial: str, llama_revised: str, mistral_initial: str, mistral_revised: str) -> str:
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


def majority_vote(choices: list[str]) -> str:
    valid_choices = [c for c in choices if c in VALID_CHOICES]
    if not valid_choices:
        return "?"
    counts = Counter(valid_choices).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return "TIE"


def all_models_agree(choices: list[str]) -> bool:
    valid = [c for c in choices if c in VALID_CHOICES]
    return len(valid) == len(MODELS) and len(set(valid)) == 1


def confidence_weighted_vote(choices: dict, confidences: dict) -> str:
    scores = {choice: 0 for choice in VALID_CHOICES}
    for model, choice in choices.items():
        if choice in VALID_CHOICES:
            scores[choice] += confidences.get(model, 50)
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return "?"
    if len([c for c, s in scores.items() if s == best_score]) > 1:
        return "TIE"
    return best


def resolve_revised_choices(question: str, revised_raw: dict, revised_choices: dict, revised_confidences: dict) -> tuple[str, str, float | None]:
    ans = majority_vote(list(revised_choices.values()))
    if ans != "TIE":
        return ans, "Revised answers had a clear majority.", 0
    try:
        start = time.time()
        raw = ask_ollama(JUDGE_MODEL, build_reasoning_judge_prompt(question, revised_raw, revised_choices, revised_confidences))
        return extract_choice(raw), raw, round(time.time() - start, 2)
    except Exception as e:
        return "?", f"ERROR: {e}", None


def tricheck_meta_decision(model_choices, model_confidences, majority_answer, confidence_weighted_answer, judge_on_disagreement_answer, judge_always_answer, reasoning_judge_answer, peer_review_answer, debate_answer, gemma_final_arbiter_answer, gemma_error_check_answer, peer_review_gemma_final_answer, debate_gemma_final_answer, option_wise_answer, gemma_centered_sequential_answer):
    method_answers = {
        "Majority Vote": majority_answer,
        "Confidence Weighted Vote": confidence_weighted_answer,
        "Judge on Disagreement": judge_on_disagreement_answer,
        "Judge Always": judge_always_answer,
        "Reasoning Judge": reasoning_judge_answer,
        "Peer Review Revision": peer_review_answer,
        "Debate Revision": debate_answer,
        "Gemma Final Arbiter": gemma_final_arbiter_answer,
        "Gemma Error-Check Revision": gemma_error_check_answer,
        "Peer Review + Gemma Final": peer_review_gemma_final_answer,
        "Debate + Gemma Final": debate_gemma_final_answer,
        "Option-wise Verifier": option_wise_answer,
        "Gemma-Centered Sequential Review": gemma_centered_sequential_answer,
    }
    model_valid = [c for c in model_choices.values() if c in VALID_CHOICES]
    if len(model_valid) == len(MODELS) and len(set(model_valid)) == 1:
        return model_valid[0], "All base models agreed."
    strong = [reasoning_judge_answer, peer_review_gemma_final_answer, debate_gemma_final_answer, option_wise_answer, gemma_centered_sequential_answer, gemma_final_arbiter_answer]
    strong_counts = Counter([x for x in strong if x in VALID_CHOICES])
    if strong_counts:
        best, count = strong_counts.most_common(1)[0]
        if count >= 3 and len([a for a, c in strong_counts.items() if c == count]) == 1:
            return best, f"Strong-system consensus: {count} votes."
    valid_methods = [a for a in method_answers.values() if a in VALID_CHOICES]
    counts = Counter(valid_methods)
    if counts:
        best, count = counts.most_common(1)[0]
        if count >= 4 and len([a for a, c in counts.items() if c == count]) == 1:
            return best, f"Meta majority among all systems: {count} votes."
    primary_choice = model_choices.get(PRIMARY_MODEL, "?")
    primary_conf = model_confidences.get(PRIMARY_MODEL, 50)
    if primary_choice in VALID_CHOICES and primary_conf >= 90:
        return primary_choice, f"Fallback to high-confidence primary model: {PRIMARY_MODEL}."
    for name, ans in [("Gemma-Centered Sequential Review", gemma_centered_sequential_answer), ("Option-wise Verifier", option_wise_answer), ("Reasoning Judge", reasoning_judge_answer), ("Gemma Final Arbiter", gemma_final_arbiter_answer), ("Peer Review + Gemma Final", peer_review_gemma_final_answer), ("Peer Review Revision", peer_review_answer), ("Majority Vote", majority_answer)]:
        if ans in VALID_CHOICES:
            return ans, f"Fallback to {name}."
    return "?", "No valid decision."


def load_questions(path: str = "questions.jsonl") -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))
    return questions[:MAX_QUESTIONS] if MAX_QUESTIONS is not None else questions


def run_revision_round(mode: str, question: str, initial_raw: dict):
    revised_raw, revised_choices, revised_conf, revised_times = {}, {}, {}, {}
    for model in MODELS:
        try:
            start = time.time()
            raw = ask_ollama(model, build_revision_prompt(model, question, initial_raw.get(model, ""), initial_raw, mode))
            revised_times[model] = round(time.time() - start, 2)
            revised_raw[model] = raw
            revised_choices[model] = extract_choice(raw)
            revised_conf[model] = extract_confidence(raw)
        except Exception as e:
            revised_raw[model] = f"ERROR: {e}"
            revised_choices[model] = "?"
            revised_conf[model] = 50
            revised_times[model] = None
    return revised_raw, revised_choices, revised_conf, revised_times


def run_gemma_call(prompt: str):
    try:
        start = time.time()
        raw = ask_ollama(PRIMARY_MODEL, prompt)
        return extract_choice(raw), raw, extract_confidence(raw), round(time.time() - start, 2)
    except Exception as e:
        return "?", f"ERROR: {e}", 50, None


def run_option_wise_verifier(question: str):
    total_scores = {choice: 0 for choice in VALID_CHOICES}
    raw_blocks, total_time = [], 0.0
    for model in MODELS:
        try:
            start = time.time()
            raw = ask_ollama(model, build_option_wise_prompt(question))
            elapsed = round(time.time() - start, 2)
            total_time += elapsed
            choice = extract_choice(raw)
            scores = parse_option_scores(raw, choice, extract_confidence(raw))
            for letter in VALID_CHOICES:
                total_scores[letter] += scores.get(letter, 0)
            raw_blocks.append(f"--- {model} ---\n{raw}\nParsed scores: {scores}\n")
        except Exception as e:
            raw_blocks.append(f"--- {model} ---\nERROR: {e}\n")
    best = max(total_scores, key=total_scores.get)
    best_score = total_scores[best]
    answer = best if best_score > 0 and len([k for k, v in total_scores.items() if v == best_score]) == 1 else "?"
    return answer, f"Aggregated option scores: {total_scores}\n\n" + "\n".join(raw_blocks), round(total_time, 2)


def run_adaptive_debate_stop(question: str, initial_choices: dict, debate_choices: dict, debate_raw: dict):
    if all_models_agree(list(initial_choices.values())):
        return list(initial_choices.values())[0], "Adaptive stop: all initial answers agreed.", 0
    debate_answer = majority_vote(list(debate_choices.values()))
    if debate_answer != "TIE":
        return debate_answer, "Adaptive stop: one debate round produced a majority.", 0
    prompt = build_gemma_final_after_revisions_prompt(question, {}, debate_raw, debate_choices, "Adaptive Debate Stop")
    answer, raw, conf, elapsed = run_gemma_call(prompt)
    return answer, raw, elapsed


def run_self_consistency_3x(question: str):
    answers, raws, total_time = [], [], 0.0
    for i in range(1, 4):
        try:
            start = time.time()
            raw = ask_ollama(PRIMARY_MODEL, build_self_consistency_prompt(question, i), temperature=0.7)
            elapsed = round(time.time() - start, 2)
            total_time += elapsed
            answer = extract_choice(raw)
            answers.append(answer)
            raws.append(f"--- sample {i} ---\n{raw}\nParsed answer: {answer}\n")
        except Exception as e:
            answers.append("?")
            raws.append(f"--- sample {i} ---\nERROR: {e}\n")
    majority = majority_vote(answers)
    if majority == "TIE":
        majority = answers[0] if answers and answers[0] in VALID_CHOICES else "?"
    return majority, f"Self-consistency samples: {answers}\n\n" + "\n".join(raws), round(total_time, 2)


def run_gemma_centered_sequential_review(question: str, initial_raw: dict):
    total_time = 0.0
    gemma_initial = initial_raw.get(PRIMARY_MODEL, "")
    llama_initial = initial_raw.get("llama3.1:8b", "")
    mistral_initial = initial_raw.get("mistral-nemo:12b", "")
    try:
        start = time.time()
        llama_revised = ask_ollama("llama3.1:8b", build_sequential_llama_revision_prompt(question, llama_initial, gemma_initial))
        total_time += round(time.time() - start, 2)
    except Exception as e:
        llama_revised = f"ERROR: {e}"
    try:
        start = time.time()
        mistral_revised = ask_ollama("mistral-nemo:12b", build_sequential_mistral_revision_prompt(question, mistral_initial, gemma_initial, llama_initial, llama_revised))
        total_time += round(time.time() - start, 2)
    except Exception as e:
        mistral_revised = f"ERROR: {e}"
    try:
        start = time.time()
        gemma_final = ask_ollama(PRIMARY_MODEL, build_sequential_gemma_final_prompt(question, gemma_initial, llama_initial, llama_revised, mistral_initial, mistral_revised))
        total_time += round(time.time() - start, 2)
    except Exception as e:
        gemma_final = f"ERROR: {e}"
    raw = f"""
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
    return extract_choice(gemma_final), raw, round(total_time, 2)


def main():
    questions = load_questions()
    rows = []
    print(f"\nTotal questions: {len(questions)}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Primary/final model: {PRIMARY_MODEL}")
    print(f"Run name: {RUN_NAME}")

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", "unknown")
        question = item["question"]
        correct_answer = item["answer"].upper()
        print("=" * 110)
        print(f"Question {index}/{len(questions)} | ID: {question_id} | {category} | {difficulty}")
        print(f"Correct answer: {correct_answer}")

        model_raw, model_choices, model_conf, model_times = {}, {}, {}, {}
        for model in MODELS:
            try:
                start = time.time()
                raw = ask_ollama(model, build_initial_prompt(question))
                elapsed = round(time.time() - start, 2)
                choice, conf = extract_choice(raw), extract_confidence(raw)
                model_raw[model], model_choices[model], model_conf[model], model_times[model] = raw, choice, conf, elapsed
                print(f"{model}: {choice} | confidence {conf} | {'✅' if choice == correct_answer else '❌'} ({elapsed} sec)")
            except Exception as e:
                model_raw[model], model_choices[model], model_conf[model], model_times[model] = f"ERROR: {e}", "?", 50, None
                print(f"{model}: ERROR {e}")

        choices = list(model_choices.values())
        majority_answer = majority_vote(choices)
        if majority_answer == "TIE": majority_answer = "?"
        confidence_weighted_answer = confidence_weighted_vote(model_choices, model_conf)
        if confidence_weighted_answer == "TIE": confidence_weighted_answer = "?"

        if all_models_agree(choices):
            judge_disagreement_answer, judge_disagreement_raw, judge_disagreement_time = choices[0], "Judge not called. All models agreed.", 0
        else:
            try:
                start = time.time()
                judge_disagreement_raw = ask_ollama(JUDGE_MODEL, build_simple_judge_prompt(question, model_choices, model_conf))
                judge_disagreement_time = round(time.time() - start, 2)
                judge_disagreement_answer = extract_choice(judge_disagreement_raw)
            except Exception as e:
                judge_disagreement_answer, judge_disagreement_raw, judge_disagreement_time = "?", f"ERROR: {e}", None

        try:
            start = time.time()
            judge_always_raw = ask_ollama(JUDGE_MODEL, build_simple_judge_prompt(question, model_choices, model_conf))
            judge_always_time = round(time.time() - start, 2)
            judge_always_answer = extract_choice(judge_always_raw)
        except Exception as e:
            judge_always_answer, judge_always_raw, judge_always_time = "?", f"ERROR: {e}", None

        try:
            start = time.time()
            reasoning_judge_raw = ask_ollama(JUDGE_MODEL, build_reasoning_judge_prompt(question, model_raw, model_choices, model_conf))
            reasoning_judge_time = round(time.time() - start, 2)
            reasoning_judge_answer = extract_choice(reasoning_judge_raw)
        except Exception as e:
            reasoning_judge_answer, reasoning_judge_raw, reasoning_judge_time = "?", f"ERROR: {e}", None

        peer_raw, peer_choices, peer_conf, peer_times = run_revision_round("peer", question, model_raw)
        peer_review_answer, peer_review_resolver_raw, peer_review_resolver_time = resolve_revised_choices(question, peer_raw, peer_choices, peer_conf)
        peer_review_time = sum(t for t in peer_times.values() if t is not None) + (peer_review_resolver_time or 0)

        debate_raw, debate_choices, debate_conf, debate_times = run_revision_round("debate", question, model_raw)
        debate_answer, debate_resolver_raw, debate_resolver_time = resolve_revised_choices(question, debate_raw, debate_choices, debate_conf)
        debate_time = sum(t for t in debate_times.values() if t is not None) + (debate_resolver_time or 0)

        gemma_final_answer, gemma_final_raw, gemma_final_conf, gemma_final_time = run_gemma_call(build_gemma_final_arbiter_prompt(question, model_raw, model_choices, model_conf))
        gemma_error_answer, gemma_error_raw, gemma_error_conf, gemma_error_time = run_gemma_call(build_gemma_error_check_prompt(question, model_raw, model_choices, model_conf))
        peer_gemma_answer, peer_gemma_raw, peer_gemma_conf, peer_gemma_time = run_gemma_call(build_gemma_final_after_revisions_prompt(question, model_raw, peer_raw, peer_choices, "Peer Review"))
        debate_gemma_answer, debate_gemma_raw, debate_gemma_conf, debate_gemma_time = run_gemma_call(build_gemma_final_after_revisions_prompt(question, model_raw, debate_raw, debate_choices, "Debate Revision"))
        option_wise_answer, option_wise_raw, option_wise_time = run_option_wise_verifier(question)
        adaptive_debate_answer, adaptive_debate_raw, adaptive_debate_time = run_adaptive_debate_stop(question, model_choices, debate_choices, debate_raw)
        self_consistency_answer, self_consistency_raw, self_consistency_time = run_self_consistency_3x(question)
        sequential_answer, sequential_raw, sequential_time = run_gemma_centered_sequential_review(question, model_raw)

        meta_answer, meta_reason = tricheck_meta_decision(model_choices, model_conf, majority_answer, confidence_weighted_answer, judge_disagreement_answer, judge_always_answer, reasoning_judge_answer, peer_review_answer, debate_answer, gemma_final_answer, gemma_error_answer, peer_gemma_answer, debate_gemma_answer, option_wise_answer, sequential_answer)

        systems_to_print = [
            ("Majority Vote", majority_answer), ("Confidence Weighted Vote", confidence_weighted_answer),
            ("Judge on Disagreement", judge_disagreement_answer), ("Judge Always", judge_always_answer),
            ("Reasoning Judge", reasoning_judge_answer), ("Peer Review Revision", peer_review_answer),
            ("Debate Revision", debate_answer), ("Gemma Final Arbiter", gemma_final_answer),
            ("Gemma Error-Check Revision", gemma_error_answer), ("Peer Review + Gemma Final", peer_gemma_answer),
            ("Debate + Gemma Final", debate_gemma_answer), ("Option-wise Verifier", option_wise_answer),
            ("Adaptive Debate Stop", adaptive_debate_answer), ("Gemma Self-Consistency 3x", self_consistency_answer),
            ("Gemma-Centered Sequential Review", sequential_answer), ("TriCheck Meta Decision", meta_answer),
        ]
        for name, ans in systems_to_print:
            extra = f" | {meta_reason}" if name == "TriCheck Meta Decision" else ""
            print(f"{name}: {ans} {'✅' if ans == correct_answer else '❌'}{extra}")

        row = {"id": question_id, "category": category, "difficulty": difficulty, "question": question, "correct_answer": correct_answer}
        for model in MODELS:
            row[f"{model}_answer"] = model_choices[model]
            row[f"{model}_confidence"] = model_conf[model]
            row[f"{model}_correct"] = model_choices[model] == correct_answer
            row[f"{model}_time_sec"] = model_times[model]
            row[f"{model}_raw"] = model_raw[model]
        fields = [
            ("majority_vote", majority_answer, None, None), ("confidence_weighted_vote", confidence_weighted_answer, None, None),
            ("judge_on_disagreement", judge_disagreement_answer, judge_disagreement_time, judge_disagreement_raw),
            ("judge_always", judge_always_answer, judge_always_time, judge_always_raw),
            ("reasoning_judge", reasoning_judge_answer, reasoning_judge_time, reasoning_judge_raw),
            ("peer_review_revision", peer_review_answer, peer_review_time, peer_review_resolver_raw),
            ("debate_revision", debate_answer, debate_time, debate_resolver_raw),
            ("gemma_final_arbiter", gemma_final_answer, gemma_final_time, gemma_final_raw),
            ("gemma_error_check_revision", gemma_error_answer, gemma_error_time, gemma_error_raw),
            ("peer_review_gemma_final", peer_gemma_answer, peer_gemma_time, peer_gemma_raw),
            ("debate_gemma_final", debate_gemma_answer, debate_gemma_time, debate_gemma_raw),
            ("option_wise_verifier", option_wise_answer, option_wise_time, option_wise_raw),
            ("adaptive_debate_stop", adaptive_debate_answer, adaptive_debate_time, adaptive_debate_raw),
            ("gemma_self_consistency_3x", self_consistency_answer, self_consistency_time, self_consistency_raw),
            ("gemma_centered_sequential_review", sequential_answer, sequential_time, sequential_raw),
            ("tricheck_meta_decision", meta_answer, None, meta_reason),
        ]
        for key, ans, t, raw in fields:
            row[f"{key}_answer"] = ans
            row[f"{key}_correct"] = ans == correct_answer
            if t is not None: row[f"{key}_time_sec"] = t
            if raw is not None: row[f"{key}_raw"] = raw
        rows.append(row)

    df = pd.DataFrame(rows)
    results_path = f"benchmark_results_{RUN_NAME}.csv"
    summary_path = f"benchmark_summary_{RUN_NAME}.csv"
    category_path = f"benchmark_by_category_{RUN_NAME}.csv"
    difficulty_path = f"benchmark_by_difficulty_{RUN_NAME}.csv"
    df.to_csv(results_path, index=False, encoding="utf-8-sig")

    systems = []
    for model in MODELS:
        systems.append({"system": model, "correct_col": f"{model}_correct", "time_col": f"{model}_time_sec"})
    systems.extend([
        {"system": "Majority Vote", "correct_col": "majority_vote_correct", "time_col": None},
        {"system": "Confidence Weighted Vote", "correct_col": "confidence_weighted_vote_correct", "time_col": None},
        {"system": "Judge on Disagreement", "correct_col": "judge_on_disagreement_correct", "time_col": "judge_on_disagreement_time_sec"},
        {"system": "Judge Always", "correct_col": "judge_always_correct", "time_col": "judge_always_time_sec"},
        {"system": "Reasoning Judge", "correct_col": "reasoning_judge_correct", "time_col": "reasoning_judge_time_sec"},
        {"system": "Peer Review Revision", "correct_col": "peer_review_revision_correct", "time_col": "peer_review_revision_time_sec"},
        {"system": "Debate Revision", "correct_col": "debate_revision_correct", "time_col": "debate_revision_time_sec"},
        {"system": "Gemma Final Arbiter", "correct_col": "gemma_final_arbiter_correct", "time_col": "gemma_final_arbiter_time_sec"},
        {"system": "Gemma Error-Check Revision", "correct_col": "gemma_error_check_revision_correct", "time_col": "gemma_error_check_revision_time_sec"},
        {"system": "Peer Review + Gemma Final", "correct_col": "peer_review_gemma_final_correct", "time_col": "peer_review_gemma_final_time_sec"},
        {"system": "Debate + Gemma Final", "correct_col": "debate_gemma_final_correct", "time_col": "debate_gemma_final_time_sec"},
        {"system": "Option-wise Verifier", "correct_col": "option_wise_verifier_correct", "time_col": "option_wise_verifier_time_sec"},
        {"system": "Adaptive Debate Stop", "correct_col": "adaptive_debate_stop_correct", "time_col": "adaptive_debate_stop_time_sec"},
        {"system": "Gemma Self-Consistency 3x", "correct_col": "gemma_self_consistency_3x_correct", "time_col": "gemma_self_consistency_3x_time_sec"},
        {"system": "Gemma-Centered Sequential Review", "correct_col": "gemma_centered_sequential_review_correct", "time_col": "gemma_centered_sequential_review_time_sec"},
        {"system": "TriCheck Meta Decision", "correct_col": "tricheck_meta_decision_correct", "time_col": None},
    ])

    summary = []
    for s in systems:
        acc = df[s["correct_col"]].mean()
        if s["time_col"] and s["time_col"] in df.columns:
            avg_time = df[s["time_col"]].dropna().mean()
            avg_time = round(avg_time, 2) if pd.notna(avg_time) else None
        else:
            avg_time = 0
        summary.append({"system": s["system"], "accuracy": round(acc * 100, 2), "avg_extra_time_sec": avg_time, "total_questions": len(df)})
    summary_df = pd.DataFrame(summary).sort_values(by="accuracy", ascending=False)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    category_rows = []
    for cat, group in df.groupby("category"):
        for s in systems:
            category_rows.append({"category": cat, "system": s["system"], "accuracy": round(group[s["correct_col"]].mean() * 100, 2), "question_count": len(group)})
    pd.DataFrame(category_rows).to_csv(category_path, index=False, encoding="utf-8-sig")

    difficulty_rows = []
    for diff, group in df.groupby("difficulty"):
        for s in systems:
            difficulty_rows.append({"difficulty": diff, "system": s["system"], "accuracy": round(group[s["correct_col"]].mean() * 100, 2), "question_count": len(group)})
    pd.DataFrame(difficulty_rows).to_csv(difficulty_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("=== SUMMARY ===")
    print(summary_df)
    print("\nFiles created:")
    print(f"- {results_path}")
    print(f"- {summary_path}")
    print(f"- {category_path}")
    print(f"- {difficulty_path}")


if __name__ == "__main__":
    main()
