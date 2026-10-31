"""Option 2: fine-tune the sentence-embedding model (all-MiniLM-L6-v2) on VIT questions.

Each training question is paired with its target text (the FAQ question it should retrieve, or
the reply text for greeting/thanks/goodbye/out-of-scope). MultipleNegativesRankingLoss pulls each
question towards its target and away from every other target in the batch, which is exactly
the job the bot's semantic search does. (An earlier attempt with BatchAllTripletLoss and its
default margin of 5 made accuracy WORSE, because normalised MiniLM vectors are at most 2 apart.)

Steps
  1. Fine-tune on the TRAIN split only, then compare base vs fine-tuned on the TEST split:
       - FAQ search accuracy  (does the closest FAQ row belong to the right FAQ? - what the bot does)
       - intent accuracy      (nearest intent centroid)
     The split-trained encoder is kept in models/_eval_encoder for the classifier's honest evaluation.
  2. Fine-tune again on ALL data -> models/vit-minilm-finetuned (used by the backend).

Outputs: training/reports/embedding_finetune_report.txt, embedding_before_after.png

Usage (from project root):  python training/finetune_embeddings.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = common.ROOT
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINAL_DIR = ROOT / "models" / "vit-minilm-finetuned"
EVAL_DIR = ROOT / "models" / "_eval_encoder"
REPORTS = ROOT / "training" / "reports"
EPOCHS = 5
BATCH = 32
LR = 5e-5


def faq_rows():
    """FAQ search rows exactly like scraper/build_index.py builds them: (text, faq_question)."""
    rows = []
    for item in common.load_faq():
        kws = item.get("keywords", [])
        variants = [s.strip() + "?" for s in item["question"].split("?") if s.strip()]
        variants.append(item["question"] + " " + " ".join(kws))
        variants += [k for k in kws if len(k.split()) >= 3]
        rows += [(v, item["question"]) for v in variants]
    return rows


def evaluate(model, train_pairs, test_pairs):
    intents = common.load_intents()
    faq_of = {it["tag"]: it.get("faq") for it in intents}
    enc = lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64)  # noqa: E731

    # FAQ search accuracy (only intents that map to a FAQ)
    # Drop FAQ rows identical to a test question so the score measures generalisation.
    test_lower = {t.lower() for t, _ in test_pairs}
    rows = [r for r in faq_rows() if r[0].lower() not in test_lower]
    row_emb = enc([r[0] for r in rows])
    faq_test = [(t, faq_of[tag]) for t, tag in test_pairs if faq_of[tag]]
    q_emb = enc([t for t, _ in faq_test])
    hits = [rows[int(np.argmax(row_emb @ q))][1] == want for q, (_, want) in zip(q_emb, faq_test)]

    # Nearest-centroid intent accuracy (all intents)
    tags = sorted({tag for _, tag in train_pairs})
    tr_emb = enc([t for t, _ in train_pairs])
    tr_tag = np.array([tag for _, tag in train_pairs])
    cent = np.stack([tr_emb[tr_tag == t].mean(0) for t in tags])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    te_emb = enc([t for t, _ in test_pairs])
    pred = [tags[i] for i in np.argmax(te_emb @ cent.T, axis=1)]
    intent_acc = np.mean([p == tag for p, (_, tag) in zip(pred, test_pairs)])
    misses = [(t, want) for (t, want), h in zip(faq_test, hits) if not h]
    return float(np.mean(hits)), float(intent_acc), misses


def target_text():
    """What each intent's questions should be pulled towards."""
    return {it["tag"]: it.get("faq") or it["response"] for it in common.load_intents()}


def finetune(pairs, out_dir: Path):
    target = target_text()
    ds = Dataset.from_dict({"anchor": [t for t, _ in pairs], "positive": [target[tag] for _, tag in pairs]})
    model = SentenceTransformer(BASE_MODEL)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir) + "_ckpt",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        learning_rate=LR,
        warmup_steps=0.1,
        # No two identical targets in a batch, so same-intent questions are never false negatives.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=common.SEED,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    losses_log = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    model.save(str(out_dir))
    shutil.rmtree(str(out_dir) + "_ckpt", ignore_errors=True)
    return model, losses_log


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    train_pairs, test_pairs = common.split()
    test_texts = [t for t, _ in test_pairs]
    train_aug = common.augment(train_pairs, exclude=test_texts)
    print(f"train {len(train_pairs)} (+aug -> {len(train_aug)}), test {len(test_pairs)}")

    base = SentenceTransformer(BASE_MODEL)
    base_faq, base_intent, base_miss = evaluate(base, train_aug, test_pairs)
    print(f"BASE       faq search {base_faq:.1%}  intent {base_intent:.1%}")

    tuned, loss_log = finetune(train_aug, EVAL_DIR)
    tuned_faq, tuned_intent, tuned_miss = evaluate(tuned, train_aug, test_pairs)
    print(f"FINE-TUNED faq search {tuned_faq:.1%}  intent {tuned_intent:.1%}")

    lines = [
        "EMBEDDING MODEL FINE-TUNING - REPORT",
        "=" * 60,
        f"Base model     : {BASE_MODEL}",
        f"Training data  : {len(train_aug)} questions ({len(train_pairs)} patterns + FAQ phrasings + typo copies)",
        f"Test data      : {len(test_pairs)} held-out questions (never used in training)",
        f"Loss           : MultipleNegativesRankingLoss (question -> its FAQ / reply text)",
        f"Training       : {EPOCHS} epochs, batch {BATCH} (no duplicate targets), lr {LR}, CPU",
        "",
        "                          BASE    FINE-TUNED",
        f"FAQ search accuracy     {base_faq:>6.1%}    {tuned_faq:>6.1%}",
        f"Intent accuracy         {base_intent:>6.1%}    {tuned_intent:>6.1%}",
        "",
        "FAQ search accuracy = for each test question, is the closest FAQ row the correct FAQ",
        "(this is exactly how the chatbot's semantic search picks an answer).",
        "",
        f"Still wrong after fine-tuning ({len(tuned_miss)}):",
        *[f"  '{t}'  -> expected: {want}" for t, want in tuned_miss],
        "",
        f"Wrong with base model ({len(base_miss)}):",
        *[f"  '{t}'" for t, _ in base_miss],
    ]
    (REPORTS / "embedding_finetune_report.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names, base_v, tuned_v = ["FAQ search", "Intent (centroid)"], [base_faq, base_intent], [tuned_faq, tuned_intent]
    x = np.arange(2)
    axes[0].bar(x - 0.18, base_v, 0.36, label="base MiniLM", color="#b9bec8")
    axes[0].bar(x + 0.18, tuned_v, 0.36, label="fine-tuned on VIT", color="#7a1f2b")
    for i in range(2):
        axes[0].text(i - 0.18, base_v[i] + 0.01, f"{base_v[i]:.0%}", ha="center")
        axes[0].text(i + 0.18, tuned_v[i] + 0.01, f"{tuned_v[i]:.0%}", ha="center")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Before vs after fine-tuning")
    axes[0].legend(loc="lower right")
    axes[1].plot(loss_log, color="#7a1f2b")
    axes[1].set_title("Fine-tuning loss")
    axes[1].set_xlabel("logging step (x10 batches)")
    fig.tight_layout()
    plt.savefig(REPORTS / "embedding_before_after.png", dpi=130)
    plt.close()

    # Final model on all data for the backend.
    all_pairs = common.examples()
    finetune(common.augment(all_pairs), FINAL_DIR)
    print(f"Saved final model -> {FINAL_DIR}")


if __name__ == "__main__":
    main()
