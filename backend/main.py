from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search import Searcher

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["searcher"] = Searcher()
    yield


app = FastAPI(title="VIT Mumbai Enquiry Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    source: str | None
    confidence: float
    matched: bool
    method: str                 # intent_model | semantic_search | website | fallback
    intent: str | None = None   # predicted intent when method == intent_model


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["searcher"].chunks)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return state["searcher"].answer(req.message.strip())
