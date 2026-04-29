import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "mistral-nemo:12b",
]

JUDGE_MODEL = "qwen3:8b"


def ask_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


def build_answer_prompt(question: str) -> str:
    return f"""
Sen bağımsız bir yapay zeka modelisin.
Aşağıdaki soruya kendi başına cevap ver.

Kurallar:
- Diğer modellerin cevaplarını bilmiyorsun.
- Emin olmadığın yerleri açıkça belirt.
- Tahmin ile kesin bilgiyi ayır.
- Cevabın sonunda 0-100 arası güven puanı ver.
- Kısa ama gerekçeli cevap ver.

Soru:
{question}
"""


def build_judge_prompt(question: str, answers: dict) -> str:
    answers_text = ""

    for model, answer in answers.items():
        answers_text += f"\n\n--- {model} cevabı ---\n{answer}\n"

    return f"""
Sen bir doğrulama ve kalite kontrol hakemisin.

Kullanıcının sorusu:
{question}

Aşağıda farklı modellerin bağımsız cevapları var:
{answers_text}

Görevin:
1. Modellerin ortak söylediği noktaları çıkar.
2. Çelişen noktaları çıkar.
3. Eksik veya şüpheli noktaları belirt.
4. En güvenilir final cevabı üret.
5. Final cevaba güven seviyesi ver: Düşük / Orta / Yüksek.

Cevabı şu başlıklarla ver:
- Ortak Noktalar
- Çelişkiler
- Şüpheli Noktalar
- Final Cevap
- Güven Seviyesi
"""


st.set_page_config(page_title="TriCheck AI", layout="wide")

st.title("TriCheck AI")
st.write("Yerel çok modelli cevap doğrulama sistemi")

question = st.text_area("Sorunu yaz:", height=150)

if st.button("Analiz Et"):
    if not question.strip():
        st.warning("Önce bir soru yaz.")
        st.stop()

    answers = {}

    with st.spinner("Modeller bağımsız cevap üretiyor..."):
        for model in MODELS:
            try:
                prompt = build_answer_prompt(question)
                answers[model] = ask_ollama(model, prompt)
            except Exception as e:
                answers[model] = f"Hata: {e}"

    st.subheader("Model Cevapları")

    cols = st.columns(len(MODELS))

    for col, model in zip(cols, MODELS):
        with col:
            st.markdown(f"### {model}")
            st.write(answers[model])

    st.subheader("Hakem Değerlendirmesi")

    with st.spinner("Hakem model cevapları karşılaştırıyor..."):
        judge_prompt = build_judge_prompt(question, answers)

        try:
            final_report = ask_ollama(JUDGE_MODEL, judge_prompt)
            st.write(final_report)
        except Exception as e:
            st.error(f"Hakem model hatası: {e}")