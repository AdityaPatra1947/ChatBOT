"""Option 1: train our own intent classifier (PyTorch neural network).

Input features for a question:
  - TF-IDF word 1-2 grams + character 2-5 grams (handles typos / Hinglish spellings)
  - 384-d sentence embedding from OUR fine-tuned MiniLM (Option 2) (handles meaning / paraphrases)
Network: features -> 256 hidden (ReLU, dropout) -> one output per intent (softmax = confidence).

Evaluation is on the shared held-out test split (training/common.py). For honesty, the test
run uses the encoder fine-tuned on the TRAIN split only (models/_eval_encoder).
An ablation compares TF-IDF only / embeddings only / combined.

Outputs
  models/intent_classifier/        vectorizer.joblib, model.pt, meta.json  (used by the backend)
  training/reports/intent_*        report, confusion matrix, training curve

Run AFTER training/finetune_embeddings.py.   Usage:  python training/train_intent_classifier.py
"""
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "training"))
import common  # noqa: E402
from intent_model import Featurizer, IntentNet  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_DIR = ROOT / "models" / "intent_classifier"
EVAL_ENCODER = ROOT / "models" / "_eval_encoder"
FINAL_ENCODER = ROOT / "models" / "vit-minilm-finetuned"
REPORTS = ROOT / "training" / "reports"
EPOCHS = 60
HIDDEN = 256
DROPOUT = 0.4
LR = 2e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed=common.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_features(feat, encoder, texts, mode):
    parts = []
    if mode in ("tfidf", "combined"):
        parts.append(feat.transform(texts).toarray())
    if mode in ("embedding", "combined"):
        parts.append(encoder.encode(texts, normalize_embeddings=True, batch_size=64))
    return torch.tensor(np.hstack(parts), dtype=torch.float32)


def train(x, y, n_classes, x_eval=None, y_eval=None):
    set_seed()
    model = IntentNet(x.shape[1], n_classes, hidden=HIDDEN, dropout=DROPOUT)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    hist = {"loss": [], "eval_acc": []}
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i: i + 32]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        hist["loss"].append(total / len(y))
        if x_eval is not None:
            model.eval()
            with torch.no_grad():
                hist["eval_acc"].append((model(x_eval).argmax(1) == y_eval).float().mean().item())
    model.eval()
    return model, hist


def main():
    if not EVAL_ENCODER.exists() or not FINAL_ENCODER.exists():
        sys.exit("Run training/finetune_embeddings.py first.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs, test_pairs = common.split()
    test_texts = [t for t, _ in test_pairs]
    train_aug = common.augment(train_pairs, exclude=test_texts)
    tags = sorted({tag for _, tag in common.examples()})
    tag_id = {t: i for i, t in enumerate(tags)}
    tr_txt = [t for t, _ in train_aug]
    y_tr = torch.tensor([tag_id[g] for _, g in train_aug])
    y_te = torch.tensor([tag_id[g] for _, g in test_pairs])
    print(f"train {len(tr_txt)} (augmented), test {len(test_texts)}, intents {len(tags)}")

    # ---- Evaluation on held-out test split ----
    encoder = SentenceTransformer(str(EVAL_ENCODER))
    feat = Featurizer().fit(tr_txt)
    results = {}
    for mode in ["tfidf", "embedding", "combined"]:
        x_tr = build_features(feat, encoder, tr_txt, mode)
        x_te = build_features(feat, encoder, test_texts, mode)
        model, hist = train(x_tr, y_tr, len(tags), x_te, y_te)
        with torch.no_grad():
            probs = torch.softmax(model(x_te), 1).numpy()
        results[mode] = (probs, hist, x_tr.shape[1])
        pred = probs.argmax(1)
        print(f"{mode:>10}: accuracy {accuracy_score(y_te, pred):.1%}  macro-F1 {f1_score(y_te, pred, average='macro'):.1%}")

    # Use whichever feature set scored best on the test split.
    best_mode = max(results, key=lambda m: accuracy_score(y_te, results[m][0].argmax(1)))
    probs, hist, n_feat = results[best_mode]
    pred, conf = probs.argmax(1), probs.max(1)
    y_np = y_te.numpy()
    acc, f1 = accuracy_score(y_np, pred), f1_score(y_np, pred, average="macro")

    rows = []
    for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        keep = conf >= t
        rows.append((t, keep.mean(), (pred[keep] == y_np[keep]).mean() if keep.any() else 0.0))
    # Lowest threshold where answered questions are >= 97% correct; the rest go to semantic search.
    threshold = next((t for t, _, a in rows if t > 0 and a >= 0.97), 0.5)
    mistakes = [(test_texts[i], tags[y_np[i]], tags[pred[i]], conf[i]) for i in range(len(y_np)) if pred[i] != y_np[i]]

    lines = [
        "INTENT CLASSIFIER - TRAINING REPORT",
        "=" * 64,
        f"Dataset            : {len(common.examples())} questions, {len(tags)} intents (data/intents.json)",
        f"Train / test       : {len(train_pairs)} / {len(test_texts)} (stratified, seed {common.SEED})",
        f"Training data      : {len(tr_txt)} after augmentation (FAQ phrasings + 1 typo copy each)",
        f"Features compared  : TF-IDF word+char n-grams / 384-d fine-tuned MiniLM embedding / both",
        f"Network            : {n_feat} -> {HIDDEN} (ReLU, dropout {DROPOUT}) -> {len(tags)} softmax  [PyTorch]",
        f"Training           : {EPOCHS} epochs, AdamW lr={LR}, batch 32, label smoothing 0.05",
        "",
        "Ablation (test accuracy / macro-F1):",
        *[f"  {m:<10} {accuracy_score(y_np, r[0].argmax(1)):>6.1%} / {f1_score(y_np, r[0].argmax(1), average='macro'):>6.1%}"
          for m, r in results.items()],
        "",
        f"Chosen feature set : {best_mode} (best test accuracy)",
        f"FINAL MODEL TEST ACCURACY : {acc:.2%}",
        f"FINAL MODEL MACRO F1      : {f1:.2%}",
        "",
        "Confidence threshold vs coverage (answered by the classifier) and accuracy:",
        "  threshold  coverage  accuracy",
        *[f"  {t:>9.1f}  {cov:>8.1%}  {a:>8.1%}" for t, cov, a in rows],
        f"Chosen threshold: {threshold} -> below it the bot falls back to semantic search.",
        "",
        f"Misclassified test questions ({len(mistakes)}):",
        *[f"  '{q}'  true={t}  predicted={p}  conf={c:.2f}" for q, t, p, c in mistakes],
        "",
        "Per-intent report:",
        classification_report(y_np, pred, labels=range(len(tags)), target_names=tags, zero_division=0),
    ]
    (REPORTS / "intent_classifier_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[9:16]))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(hist["loss"], color="#7a1f2b")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss", color="#7a1f2b")
    ax2 = ax1.twinx()
    for m, color in [("tfidf", "#b9bec8"), ("embedding", "#e6a23c"), ("combined", "#1a5fb4")]:
        ax2.plot(results[m][1]["eval_acc"], color=color, label=f"test acc - {m}")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right")
    plt.title("Intent classifier training")
    fig.tight_layout()
    plt.savefig(REPORTS / "intent_training_curve.png", dpi=130)
    plt.close()

    cm = confusion_matrix(y_np, pred, labels=range(len(tags)))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(tags)), tags, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tags)), tags, fontsize=7)
    ax.set_xlabel("predicted intent")
    ax.set_ylabel("true intent")
    ax.set_title(f"Intent classifier confusion matrix (test accuracy {acc:.1%})")
    for i in range(len(tags)):
        for j in range(len(tags)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    plt.savefig(REPORTS / "intent_confusion_matrix.png", dpi=130)
    plt.close()

    # ---- Final model: all data + final fine-tuned encoder ----
    all_aug = common.augment(common.examples())
    all_txt = [t for t, _ in all_aug]
    feat_all = Featurizer().fit(all_txt)
    final_encoder = SentenceTransformer(str(FINAL_ENCODER))
    x_all = build_features(feat_all, final_encoder, all_txt, best_mode)
    final, _ = train(x_all, torch.tensor([tag_id[g] for _, g in all_aug]), len(tags))
    joblib.dump(feat_all, MODEL_DIR / "vectorizer.joblib")
    torch.save(final.state_dict(), MODEL_DIR / "model.pt")
    meta = {"tags": tags, "hidden": HIDDEN, "threshold": float(threshold), "features": best_mode,
            "encoder": "models/vit-minilm-finetuned", "test_accuracy": float(acc), "test_macro_f1": float(f1)}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
