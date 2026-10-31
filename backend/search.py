"""Answer pipeline: trained intent classifier -> semantic search over FAQ/documents -> fallback."""
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from intent_model import IntentClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FINETUNED = ROOT / "models" / "vit-minilm-finetuned"
# Our fine-tuned encoder if it has been trained, else the base model.
MODEL = str(FINETUNED) if FINETUNED.exists() else "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "models/vit-minilm-finetuned" if FINETUNED.exists() else MODEL

FAQ_BOOST = 0.10       # curated answers win close ties against scraped text
FAQ_THRESHOLD = 0.55   # minimum raw score for a FAQ answer
DOC_THRESHOLD = 0.48   # minimum raw score for a document answer
DOC_PASSAGES = 3       # max passages combined from the best page
DOC_SIMILAR = 0.08     # extra passages must score within this of the best one
ANSWER_CHARS = 1100
FALLBACK = (
    "Sorry, I couldn't find that in my VIT information.\n\n"
    "You can ask me about admissions, fees, courses, placements, internships, hostel, "
    "scholarships or campus facilities. For anything else, contact the VIT office on "
    "**022-2416 1126** (Mon-Sat, 9 am to 5 pm)."
)


def tidy(text: str) -> str:
    """Clean OCR/table artefacts and trim to whole sentences."""
    text = re.sub(r"\s*\|\s*", ", ", text)
    text = re.sub(r"(\s*\.\s*){2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"[.?!]\s+(?=[A-Z])", text[:200])
    if m:
        text = text[m.end():]
    if len(text) > ANSWER_CHARS:
        cut = text[:ANSWER_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "))
        text = cut[: end + 1] if end > 300 else cut.rsplit(" ", 1)[0] + "..."
    return text


def page_name(chunk: dict) -> str:
    title = chunk.get("title") or ""
    if title.lower().endswith(".pdf"):
        title = re.sub(r"[-_]+", " ", title[:-4])
    title = re.sub(r"^VIT\s*\|\s*", "", title)
    title = re.sub(r"\s*[-|]\s*VIT\s*\|\s*Vidyalankar Institute of Technology$", "", title).strip()
    return title or "the VIT website"


class Searcher:
    def __init__(self):
        self.model = SentenceTransformer(MODEL)
        self.chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
        self.emb = np.load(DATA / "embeddings.npy")
        self.is_faq = np.array([c["kind"] == "faq" for c in self.chunks])
        meta_path = DATA / "index_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if meta.get("model", MODEL_ID) != MODEL_ID:
            raise RuntimeError("Index was built with a different model - run scraper/build_index.py")

        self.classifier = IntentClassifier() if IntentClassifier.available() else None
        faq_by_question = {f["question"]: f for f in json.loads((DATA / "faq.json").read_text(encoding="utf-8"))}
        self.intents = {}
        for it in json.loads((DATA / "intents.json").read_text(encoding="utf-8"))["intents"]:
            faq = faq_by_question.get(it.get("faq"))
            self.intents[it["tag"]] = {
                "answer": faq["answer"] if faq else it["response"],
                "source": faq["source"] if faq else None,
            }

    def _doc_answer(self, raw: np.ndarray) -> dict | None:
        doc_scores = np.where(self.is_faq, -1.0, raw)
        top = np.argsort(-doc_scores)[:15]
        best = int(top[0])
        best_score = float(doc_scores[best])
        if best_score < DOC_THRESHOLD:
            return None
        source = self.chunks[best]["source"]
        picked = [int(i) for i in top
                  if self.chunks[int(i)]["source"] == source
                  and doc_scores[int(i)] >= best_score - DOC_SIMILAR][:DOC_PASSAGES]
        body = tidy(" ".join(self.chunks[i]["text"] for i in sorted(picked)))
        answer = f"Here's what I found in **{page_name(self.chunks[best])}**:\n\n{body}"
        return {"answer": answer, "source": source, "confidence": best_score, "matched": True,
                "method": "website"}

    def answer(self, question: str) -> dict:
        q = self.model.encode([question], normalize_embeddings=True)[0]

        # 1. Our trained intent classifier, when it is confident.
        if self.classifier:
            tag, conf = self.classifier.predict(question, q)
            if conf >= self.classifier.threshold and tag in self.intents:
                it = self.intents[tag]
                return {"answer": it["answer"], "source": it["source"], "confidence": conf,
                        "matched": tag != "out_of_scope", "method": "intent_model", "intent": tag}

        # 2. Semantic search over FAQ rows (fine-tuned embeddings), then website passages.
        raw = self.emb @ q
        best = int(np.argmax(raw + FAQ_BOOST * self.is_faq))
        chunk, score = self.chunks[best], float(raw[best])
        if chunk["kind"] == "faq" and score >= FAQ_THRESHOLD:
            return {"answer": chunk["text"], "source": chunk["source"], "confidence": score,
                    "matched": True, "method": "semantic_search"}
        doc = self._doc_answer(raw)
        if doc:
            return doc
        return {"answer": FALLBACK, "source": None, "confidence": score, "matched": False, "method": "fallback"}
