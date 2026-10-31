# VIT Mumbai Enquiry Chatbot

Offline enquiry chatbot for Vidyalankar Institute of Technology, Wadala, Mumbai.
FAQ + semantic search (sentence-transformers `all-MiniLM-L6-v2`), FastAPI backend, React (Vite) frontend.
Runs locally only - no paid API, no deployment.

See [PLAN.md](PLAN.md) for the design and [SITES.md](SITES.md) for which VIT sites were scraped vs. need credentials.

## Project layout
```
data/faq.json          curated Q&A with full answers (edit this to improve answers)
data/manual/           drop extra .txt/.md/.pdf files here (e.g. info from login-only portals)
data/raw/pages.json    scraped public pages (generated)
data/chunks.json, data/embeddings.npy   search index (generated)
data/embed_cache.npz   embedding cache so rebuilds only embed changed text (generated)
scraper/scrape.py      crawls public vit.edu.in pages + PDFs (robots.txt, 1s delay),
                       OCRs scanned PDFs, skips merit lists / student lists (privacy)
scraper/build_index.py chunks + embeds FAQ, scraped pages and data/manual
backend/main.py        FastAPI: POST /chat, GET /health
backend/search.py      semantic search, thresholds, fallback
backend/eval_questions.py  43 sample questions quality check
frontend/              React chat UI
```

## Setup (one time)
```bash
pip install -r backend/requirements.txt
```
```bash
npm install --prefix frontend
```

## Refresh data (optional - index is already built)
```bash
python scraper/scrape.py
```
```bash
python scraper/build_index.py
```
Re-run `build_index.py` after any edit to `data/faq.json` or `data/manual/`.

## Run
Terminal 1 (backend, http://localhost:8000):
```bash
cd backend && python -m uvicorn main:app --port 8000
```
Terminal 2 (frontend, http://localhost:5173):
```bash
npm run dev --prefix frontend
```
Open http://localhost:5173.

## Check answer quality
```bash
cd backend && python eval_questions.py
```

## Tuning
In `backend/search.py`:
- `FAQ_THRESHOLD` / `DOC_THRESHOLD` - raise to reduce wrong answers, lower to answer more.
- `FAQ_BOOST` - how strongly curated FAQ answers are preferred over scraped text.

## Trained models
Two models are trained on VIT data (CPU is enough, about 10 minutes total):

| Model | What it learns | Script | Output |
|---|---|---|---|
| **Intent classifier** (Option 1) | Maps a question to one of 51 intents (47 FAQ topics + greeting, thanks, goodbye, out-of-scope). PyTorch network on TF-IDF word/char n-grams + fine-tuned sentence embeddings. | `training/train_intent_classifier.py` | `models/intent_classifier/` |
| **Fine-tuned MiniLM** (Option 2) | `all-MiniLM-L6-v2` fine-tuned so VIT questions land next to the right FAQ (MultipleNegativesRankingLoss). Used by the classifier and by semantic search. | `training/finetune_embeddings.py` | `models/vit-minilm-finetuned/` |

Training data: `data/intents.json` (943 example questions incl. Hinglish and typos) + FAQ phrasings + typo augmentation. Both scripts share one stratified 80/20 split (`training/common.py`), so reported scores are on questions never seen in training.

Retrain (in this order), then rebuild the index:
```bash
pip install -r training/requirements.txt
```
```bash
python training/finetune_embeddings.py
```
```bash
python training/train_intent_classifier.py
```
```bash
python scraper/build_index.py
```
Reports (accuracy, ablation, confusion matrix, curves, before/after chart) are written to `training/reports/`.

## How answers work
1. **Trained intent model**: if the classifier is confident (above the threshold chosen during training), the bot returns that intent's written answer from `data/faq.json` (or the greeting/thanks/out-of-scope reply).
2. **Semantic search**: otherwise the fine-tuned MiniLM finds the closest FAQ.
3. **Website passage**: if no FAQ is close, it combines up to 3 relevant passages from the best-matching scraped page and says which page they came from.
4. **Fallback**: otherwise it suggests topics and gives the office number.

Each reply shows which step answered and its confidence (e.g. "Trained intent model · 94%").

The chat UI renders `**bold**`, `• ` bullet lists and `1. ` numbered lists, and shows a small "View official source" link under each answer.

To teach a new question: add an entry to `data/faq.json` (question, answer, keywords incl. a few full phrasings like "mtech fees", source), then rebuild the index. Short ambiguous words (e.g. "mtech" vs "btech") need explicit full-phrase keywords.

## Known limitations
- Scanned PDFs are read with OCR (RapidOCR); table columns can come out scrambled, which is why key fee tables are written into `faq.json` by hand from the official notices.
- Website answers are extracted passages (no text generation), so they can read less smoothly than FAQ answers.
- Fees, dates, HoDs and hostel info change yearly - re-scrape and review `faq.json` each admission season. Current FAQ figures are for A.Y. 2026-27.
