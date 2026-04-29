# TriCheck AI

TriCheck AI is a local multi-model LLM verification and benchmarking system built with Ollama.

The project compares multiple local LLM decision strategies on a 100-question MMLU subset.

## Models

- gemma3:12b
- llama3.1:8b
- mistral-nemo:12b

## Evaluated Systems

- Single-model baseline
- Majority Vote
- Confidence Weighted Vote
- Judge on Disagreement
- Judge Always
- Reasoning Judge
- Peer Review Revision
- Debate Revision
- Gemma Final Arbiter
- Gemma Self-Consistency 3x
- Gemma-Centered Sequential Review
- Coarse Pattern-Best Router

## Final Result Summary

On a 100-question MMLU subset:

| System | Accuracy |
|---|---:|
| Coarse Pattern-Best Router | 69% |
| Judge Always | 68% |
| Gemma Self-Consistency 3x | 68% |
| Judge on Disagreement | 67% |
| Gemma Final Arbiter | 66% |
| gemma3:12b | 63% |
| Majority Vote | 58% |
| mistral-nemo:12b | 58% |
| llama3.1:8b | 56% |

## Key Finding

Naive majority voting was weaker than the strongest single model.  
The best result came from a pattern-based router, improving over:

- Gemma3:12B by +6 points
- Majority Vote by +11 points

This suggests that local LLM ensemble performance depends more on selecting the right decision strategy than simply adding more models.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt