"""Intent classifier shared by training (training/train_intent_classifier.py) and the backend."""
import json
from pathlib import Path

import joblib
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "intent_classifier"


class Featurizer:
    """Word 1-2 gram + character 2-5 gram TF-IDF, so typos and Hinglish still match."""

    def __init__(self):
        self.word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self.char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)

    def fit(self, texts):
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    @property
    def n_features(self):
        return len(self.word.vocabulary_) + len(self.char.vocabulary_)


class IntentNet(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def to_tensor(sparse):
    return torch.tensor(sparse.toarray(), dtype=torch.float32)


class IntentClassifier:
    """Loads the trained model; predict() returns (tag, confidence).

    If the model uses sentence embeddings, pass the question's normalised embedding
    (from the fine-tuned MiniLM) to predict().
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.tags = meta["tags"]
        self.threshold = meta["threshold"]
        # "tfidf", "embedding" or "combined" - whichever won during training.
        self.features = meta.get("features", "tfidf")
        self.test_accuracy = meta.get("test_accuracy")
        self.featurizer = joblib.load(model_dir / "vectorizer.joblib")
        state = torch.load(model_dir / "model.pt", weights_only=True)
        n_in = state["net.1.weight"].shape[1]
        self.model = IntentNet(n_in, len(self.tags), hidden=meta["hidden"])
        self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def available(model_dir: Path = MODEL_DIR) -> bool:
        return (model_dir / "meta.json").exists()

    def predict(self, text: str, embedding=None) -> tuple[str, float]:
        parts = []
        if self.features in ("tfidf", "combined"):
            parts.append(to_tensor(self.featurizer.transform([text])))
        if self.features in ("embedding", "combined"):
            parts.append(torch.tensor(embedding, dtype=torch.float32).reshape(1, -1))
        x = torch.cat(parts, dim=1)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1)[0]
        i = int(probs.argmax())
        return self.tags[i], float(probs[i])
