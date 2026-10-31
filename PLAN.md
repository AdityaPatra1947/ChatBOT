# VIT Mumbai Enquiry Chatbot - Project Plan

Chatbot that answers admission and general enquiries for Vidyalankar Institute of Technology (VIT), Wadala Road, Mumbai.

## Decisions
- Approach: FAQ + semantic search (embeddings, offline, no paid API)
- Backend: FastAPI (Python)
- Frontend: React (Vite)
- Data: scraped from public VIT Mumbai websites + a curated `faq.json`
- Scope: build and run locally only. No deployment.

## Architecture
```
React chat UI  ->  FastAPI /chat  ->  Embed question (sentence-transformers)
                                      -> cosine search over FAQ + document chunks
                                      -> best match above threshold -> answer + source
                                      -> else fallback (contact admission office)
```

## Folder Structure
```
AI Project/
  PLAN.md
  SITES.md                 # list of sites, public vs login-required
  data/
    raw/                   # scraped pages (html/text)
    faq.json               # curated question/answer pairs
    chunks.json            # cleaned document chunks
    embeddings.npy         # precomputed vectors
  scraper/
    scrape.py              # fetch + clean public pages
    build_index.py         # chunk + embed -> embeddings.npy
  backend/
    main.py                # FastAPI app, /chat and /health
    search.py              # semantic search + fallback
    requirements.txt
  frontend/                # React (Vite) chat UI
    src/
  README.md
```

## Phases
1. **Site inventory** - list all VIT Mumbai sites; mark public vs credential-protected (`SITES.md`).
2. **Data collection** - scrape public pages; user supplies content from login-only sites.
3. **Data preparation** - clean text, split into chunks, write `faq.json`.
4. **Indexing** - embed FAQ + chunks with `all-MiniLM-L6-v2`, save vectors.
5. **Backend** - FastAPI `/chat` returning answer, confidence, source; CORS for React.
6. **Frontend** - chat window, quick-reply buttons (Admissions, Fees, Courses, Placements, Contact), typing indicator, source links.
7. **Testing** - 30-50 sample student/parent questions; tune similarity threshold.
8. **Docs** - README with local run steps.

## Topics Covered
Courses and branches, admission process and eligibility, fees and scholarships, cutoffs, placements, campus facilities, departments, contact and location, important dates, fallback to admission office.

## Notes
- Scraping respects robots.txt and uses a polite delay.
- Login-protected portals (ERP, LMS, student/faculty) are NOT scraped by me; content from them is provided by the user as exported files.
- Fees and dates change yearly; each answer carries a "last updated" note.
