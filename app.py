"""Streamlit demo — Vietnamese RAG over the AI/ML Wikipedia corpus.

This app is deliberately thin: it imports the SAME `retrieval.py` and `rag_qa.py`
the evaluation uses, so what a reviewer clicks here is what the README measured.
A demo that reimplements the pipeline is a second implementation, and second
implementations drift.

Run:
    ollama serve                     # local model server (models on D:)
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import streamlit as st

from rag_qa import DEFAULT_MODEL, answer
from retrieval import Retriever

st.set_page_config(page_title="Hỏi đáp tài liệu AI/ML tiếng Việt", page_icon="📚", layout="wide")

EXAMPLES = [
    ("Trong corpus", "Chưng cất tri thức là gì?"),
    ("Trong corpus", "So sánh học có giám sát và học không có giám sát"),
    ("Trong corpus", "Mạng nơ-ron tích chập và mạng nơ-ron hồi quy khác nhau thế nào?"),
    ("Paraphrase", "Vì sao mô hình đạt kết quả rất tốt trên dữ liệu huấn luyện nhưng lại kém khi gặp dữ liệu mới?"),
    ("Ngoài corpus", "Cách nấu phở bò ngon tại nhà?"),
]


@st.cache_resource(show_spinner="Đang nạp 4.133 đoạn văn và model embedding (một lần duy nhất)…")
def get_retriever() -> Retriever:
    return Retriever()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    k = st.slider("Số đoạn đưa vào prompt", 3, 10, 5,
                  help="5 là giá trị đo được tốt nhất: 9/12 câu trích dẫn đúng bài gold "
                       "(k=10 chỉ đạt 6/12 — nhiều đoạn gây nhiễu làm model trích dẫn sai).")
    model = st.text_input("Model (Ollama)", DEFAULT_MODEL)
    show_prompt = st.checkbox("Hiện prompt đầy đủ", value=False)

    st.divider()
    st.caption(
        "**Pipeline**\n\n"
        "1. BM25 + dense (MiniLM multilingual)\n"
        "2. Gộp thứ hạng bằng RRF\n"
        "3. Luôn nhận top-2 của mỗi retriever\n"
        "4. LLM trả lời **chỉ từ** các đoạn trên, bắt buộc trích dẫn"
    )
    st.caption("Chỉ số retrieval: Recall@10 **0.958** · MRR **0.817** · nDCG@10 **0.763**")

# ── main ──────────────────────────────────────────────────────────────────────
st.title("📚 Hỏi đáp tài liệu AI/ML tiếng Việt")
st.caption(
    "RAG chạy **hoàn toàn local** trên 179 bài Wikipedia tiếng Việt (4.133 đoạn văn). "
    "Model từ chối trả lời khi tài liệu không chứa câu trả lời — thà nói không biết còn hơn bịa."
)

if "query" not in st.session_state:
    # deep link support: app/?q=câu+hỏi — makes a given question shareable, and
    # makes any screenshot in the README reproducible from its URL.
    st.session_state.query = st.query_params.get("q", "")

st.write("**Câu hỏi mẫu:**")
cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES):
    if col.button(f"{label}\n\n{q[:38]}…", use_container_width=True):
        st.session_state.query = q

query = st.text_input("Câu hỏi của bạn", value=st.session_state.query,
                      placeholder="ví dụ: Học máy lượng tử là gì?")

if query:
    try:
        retriever = get_retriever()
    except Exception as exc:  # noqa: BLE001 — surface the real error, don't hide it
        st.error(f"Không nạp được index: {exc}")
        st.stop()

    with st.spinner(f"Đang tìm kiếm và sinh câu trả lời bằng {model}…"):
        try:
            result = answer(query, retriever=retriever, k=k, model=model)
        except SystemExit as exc:
            st.error(str(exc))
            st.stop()

    audit = result["audit"]

    # ── answer ────────────────────────────────────────────────────────────────
    if audit["refused"]:
        st.warning(result["answer"])
        st.caption("Model từ chối vì các đoạn tìm được không chứa câu trả lời.")
    else:
        st.success(result["answer"])

    # ── audit row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trích dẫn", ", ".join(f"[{i}]" for i in audit["cited"]) or "—")
    c2.metric("Trích dẫn hợp lệ", "✅" if not audit["invalid"] else f"❌ {audit['invalid']}")
    c3.metric("Từ chối trả lời", "có" if audit["refused"] else "không")
    c4.metric("Nguồn trong context", len(result["contexts"]))

    # ── sources ───────────────────────────────────────────────────────────────
    with st.expander(f"📄 {len(result['contexts'])} đoạn tài liệu đã đưa vào prompt", expanded=True):
        for i, ctx in enumerate(result["contexts"], 1):
            cited = " ✅ được trích dẫn" if i in audit["cited"] else ""
            st.markdown(f"**[{i}] {ctx['title']}** — chunk `{ctx['chunk_id']}`{cited}")
            st.caption(ctx["text"])
            st.markdown(f"[Xem bài gốc trên Wikipedia](https://vi.wikipedia.org/?curid={ctx['doc_id']})")
            st.divider()

    if show_prompt:
        with st.expander("🔍 Prompt đầy đủ gửi cho model"):
            st.code(result["prompt"], language="text")
else:
    st.info("Nhập câu hỏi hoặc bấm một câu hỏi mẫu ở trên để bắt đầu.")

st.divider()
st.caption(
    "Demo này gọi thẳng `scripts/retrieval.py` và `scripts/rag_qa.py` — cùng đoạn code mà "
    "bảng kết quả trong README được đo bằng. Corpus: Wikipedia tiếng Việt, CC BY-SA 4.0."
)
