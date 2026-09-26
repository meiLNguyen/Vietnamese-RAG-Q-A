"""RAG Q&A: hybrid retrieval + local LLM generation, with citations.

Design rules (see SCOPE.md):
  * the generator may ONLY use the retrieved passages — no outside knowledge
  * every claim must carry a citation marker [1], [2], ... pointing at a passage
  * if the passages do not contain the answer, the model must refuse — a
    confident hallucination is worse than an honest "I don't know"

Runs fully local through Ollama. Start the server first:
    ollama serve            # models live on D: (OLLAMA_MODELS)

Usage:
    python scripts/rag_qa.py "Chưng cất tri thức là gì?"
    python scripts/rag_qa.py --k 5 "So sánh học có giám sát và học không có giám sát"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import Retriever

OLLAMA_CHAT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"
REFUSAL = "Tôi không tìm thấy thông tin trong tài liệu được cung cấp."

# ── Prompt ────────────────────────────────────────────────────────────────────
# Rule 1 stops the model from answering out of its own memory.
# Rule 2 makes the answer auditable: every claim must be traceable to a passage.
# Rule 3 gives the model an explicit escape hatch, which it will not invent one for.
# Rule 4 keeps answers in the language and shape the user expects.
PROMPT_TEMPLATE = """Bạn là trợ lý trả lời câu hỏi CHỈ dựa vào các đoạn tài liệu được cung cấp.

QUY TẮC BẮT BUỘC:
1. Chỉ dùng thông tin trong phần TÀI LIỆU bên dưới. Không dùng kiến thức bên ngoài.
2. Mỗi khẳng định phải kèm trích dẫn số hiệu tài liệu, dạng [1] hoặc [2][3].
3. Nếu TÀI LIỆU không chứa câu trả lời, trả lời đúng một câu: "{refusal}" — TUYỆT ĐỐI không bịa.
4. Trả lời ngắn gọn, trực tiếp, bằng tiếng Việt.

TÀI LIỆU:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""


def build_prompt(query: str, contexts: list[dict]) -> str:
    """Render the prompt with numbered passages (the numbers are the citation ids)."""
    blocks = [f"[{i}] ({c['title']})\n{c['text']}" for i, c in enumerate(contexts, start=1)]
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), query=query, refusal=REFUSAL)


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    """Single-turn call to the local Ollama server. Low temperature: we want grounded, not creative."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 4096},
    }
    req = urllib.request.Request(OLLAMA_CHAT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["message"]["content"].strip()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Không gọi được Ollama tại {OLLAMA_CHAT} — đã chạy `ollama serve` chưa?\n{exc}")


def check_citations(answer: str, n_contexts: int) -> dict:
    """Audit citation markers: which passages were cited, and were they real?

    An answer citing [7] when only 5 passages exist is not a style problem —
    it is an unsupported claim, and it must be detectable without a human reading.
    """
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    invalid = [c for c in cited if c < 1 or c > n_contexts]
    return {"cited": cited, "invalid": invalid, "has_citation": bool(cited),
            "refused": REFUSAL in answer}


def answer(query: str, retriever: Retriever | None = None, k: int = 5,
           model: str = DEFAULT_MODEL, verbose: bool = False) -> dict:
    """Full pipeline: retrieve -> prompt -> generate -> audit."""
    retriever = retriever or Retriever()
    contexts = retriever.retrieve(query, k=k)
    prompt = build_prompt(query, contexts)
    text = ask_ollama(prompt, model=model)
    audit = check_citations(text, len(contexts))
    if verbose:
        print("── passages ──")
        for i, c in enumerate(contexts, 1):
            print(f"[{i}] {c['title']} ({c['chunk_id']}) {c['text'][:90]}...")
        print("\n── prompt ──")
        print(prompt)
        print("\n── answer ──")
    return {"query": query, "answer": text, "contexts": contexts, "audit": audit, "prompt": prompt}


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG Q&A over the Vietnamese corpus (local, via Ollama).")
    ap.add_argument("query", help="câu hỏi")
    ap.add_argument("--k", type=int, default=5, help="số đoạn đưa vào prompt (default 5 — đo được là tốt nhất)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verbose", action="store_true", help="in cả passages và prompt")
    args = ap.parse_args()

    result = answer(args.query, k=args.k, model=args.model, verbose=args.verbose)
    a = result["audit"]
    print(result["answer"])
    print(f"\n--- audit: trích dẫn {a['cited']} · hợp lệ? {'✅' if not a['invalid'] else '❌ ' + str(a['invalid'])}"
          f" · có trích dẫn? {'✅' if a['has_citation'] else '❌'} · từ chối? {a['refused']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
