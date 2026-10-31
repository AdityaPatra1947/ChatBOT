"""Chunk scraped pages + faq.json + data/manual files, embed, save index.

Run after scrape.py. Output: data/chunks.json, data/embeddings.npy
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FINETUNED = ROOT / "models" / "vit-minilm-finetuned"
# Same rule as backend/search.py: our fine-tuned encoder if trained, else the base model.
MODEL = str(FINETUNED) if FINETUNED.exists() else "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "models/vit-minilm-finetuned" if FINETUNED.exists() else MODEL
CHUNK_WORDS, OVERLAP = 120, 25


def split(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i : i + CHUNK_WORDS])
        if len(piece.split()) >= 15:
            yield piece


def main():
    chunks = []

    faq_path = DATA / "faq.json"
    if faq_path.exists():
        for item in json.loads(faq_path.read_text(encoding="utf-8")):
            # One row per phrasing so short queries still match: each question sentence,
            # the question + all keywords, and multi-word keyword phrases. Single-word
            # keywords alone are too generic ("btech", "mumbai") and cause false hits.
            keywords = item.get("keywords", [])
            variants = [s.strip() + "?" for s in item["question"].split("?") if s.strip()]
            variants.append(item["question"] + " " + " ".join(keywords))
            variants += [k for k in keywords if len(k.split()) >= 3]
            for v in variants:
                chunks.append({
                    "kind": "faq",
                    "question": item["question"],
                    "text": item["answer"],
                    "source": item.get("source", "https://vit.edu.in"),
                    "title": "FAQ",
                    "embed_text": v,
                })

    pages_path = DATA / "raw" / "pages.json"
    if pages_path.exists():
        for p in json.loads(pages_path.read_text(encoding="utf-8")):
            for piece in split(p["text"]):
                chunks.append({"kind": "doc", "text": piece, "source": p["url"],
                               "title": p["title"], "embed_text": p["title"] + ". " + piece})

    manual = DATA / "manual"
    for f in manual.glob("*") if manual.exists() else []:
        if f.suffix.lower() in {".txt", ".md"}:
            text = f.read_text(encoding="utf-8", errors="ignore")
        elif f.suffix.lower() == ".pdf":
            text = " ".join((pg.extract_text() or "") for pg in PdfReader(f).pages)
        else:
            continue
        for piece in split(re.sub(r"\s+", " ", text)):
            chunks.append({"kind": "doc", "text": piece, "source": f"manual:{f.name}",
                           "title": f.name, "embed_text": piece})

    # Cache vectors by text hash so editing faq.json only re-embeds what changed.
    cache_path = DATA / "embed_cache.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    # Key includes the model (and its weights' timestamp), so vectors from different or
    # retrained models never mix.
    weights = Path(MODEL) / "model.safetensors"
    model_tag = Path(MODEL).name + (f"@{weights.stat().st_mtime_ns}" if weights.exists() else "")
    keys = [hashlib.sha1((model_tag + "|" + c["embed_text"]).encode("utf-8")).hexdigest() for c in chunks]
    missing = sorted({k: c["embed_text"] for k, c in zip(keys, chunks) if k not in cache}.items())
    print(f"{len(chunks)} chunks; embedding {len(missing)} new...")
    if missing:
        model = SentenceTransformer(MODEL)
        vecs = model.encode([t for _, t in missing], normalize_embeddings=True,
                            show_progress_bar=True, batch_size=64)
        cache.update({k: v for (k, _), v in zip(missing, vecs)})
    emb = np.stack([cache[k] for k in keys])
    np.savez(cache_path, **{k: cache[k] for k in set(keys)})
    for c in chunks:
        c.pop("embed_text")
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(DATA / "embeddings.npy", emb.astype("float32"))
    (DATA / "index_meta.json").write_text(json.dumps({"model": MODEL_ID}), encoding="utf-8")
    print("Index saved.")


if __name__ == "__main__":
    main()
