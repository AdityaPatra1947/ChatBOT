"""Shared data loading, train/test split and augmentation for both training scripts.

Both scripts use the SAME split (stratified, fixed seed), so test questions are never seen
by either model during evaluation.
"""
import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TEST_SIZE = 0.2


def load_intents():
    return json.loads((ROOT / "data" / "intents.json").read_text(encoding="utf-8"))["intents"]


def load_faq():
    return json.loads((ROOT / "data" / "faq.json").read_text(encoding="utf-8"))


def examples():
    """(text, tag) pairs from intents.json patterns."""
    return [(p, it["tag"]) for it in load_intents() for p in it["patterns"]]


def split():
    """Deterministic stratified split -> (train_pairs, test_pairs)."""
    pairs = examples()
    texts, tags = zip(*pairs)
    tr_x, te_x, tr_y, te_y = train_test_split(
        list(texts), list(tags), test_size=TEST_SIZE, stratify=list(tags), random_state=SEED
    )
    return list(zip(tr_x, tr_y)), list(zip(te_x, te_y))


def faq_phrasings():
    """Extra training text: each FAQ's question sentences and multi-word keywords, labelled with its intent."""
    by_question = {it["faq"]: it["tag"] for it in load_intents() if "faq" in it}
    out = []
    for item in load_faq():
        tag = by_question.get(item["question"])
        if not tag:
            continue
        out += [(s.strip() + "?", tag) for s in item["question"].split("?") if s.strip()]
        out += [(k, tag) for k in item.get("keywords", []) if len(k.split()) >= 2]
    return out


def typo(text: str, rng: random.Random) -> str:
    """One random keyboard-style typo: drop, swap or duplicate a character."""
    if len(text) < 5:
        return text
    i = rng.randrange(1, len(text) - 1)
    op = rng.choice(["drop", "swap", "dup"])
    if op == "drop":
        return text[:i] + text[i + 1:]
    if op == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[:i] + text[i] + text[i:]


def augment(pairs, copies=1, seed=SEED, exclude=()):
    """Original pairs + FAQ phrasings + `copies` typo variants of each pattern (training data only).

    `exclude`: texts that must not appear in the output (the test questions), to avoid leakage.
    """
    rng = random.Random(seed)
    banned = {t.lower() for t in exclude}
    base = list(pairs) + [(t, tag) for t, tag in faq_phrasings() if t.lower() not in banned]
    noisy = [(typo(t, rng), tag) for t, tag in pairs for _ in range(copies)]
    seen, out = set(), []
    for t, tag in base + noisy:
        if (t.lower(), tag) not in seen and t.lower() not in banned:
            seen.add((t.lower(), tag))
            out.append((t, tag))
    return out
