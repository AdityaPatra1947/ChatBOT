import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const QUICK = [
  "How do I get admission to B.Tech?",
  "What is the fee structure?",
  "Documents required for admission",
  "What courses does VIT offer?",
  "Tell me about placements",
  "Is there a hostel?",
  "Are scholarships available?",
  "Where is VIT located?",
];

// Which part of the pipeline produced the answer (backend/search.py).
const METHOD_LABEL = {
  intent_model: "Trained intent model",
  semantic_search: "Semantic search",
  website: "Website passage",
  fallback: "No match",
};

const WELCOME = {
  role: "bot",
  text: "Hi! I'm the VIT Mumbai enquiry assistant.\n\nAsk me about **admissions, fees, documents, courses, placements, internships, hostel, scholarships** or campus life.",
};

// Inline formatting: **bold** and bare URLs.
function inline(text, keyBase) {
  return text.split(/(\*\*[^*]+\*\*|https?:\/\/[^\s)]+)/g).map((part, i) => {
    const key = `${keyBase}-${i}`;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (/^https?:\/\//.test(part)) {
      const url = part.replace(/[.,]$/, "");
      return (
        <a key={key} href={url} target="_blank" rel="noreferrer">
          {url}
        </a>
      );
    }
    return part;
  });
}

// Block formatting: paragraphs, "• " bullet lists and "1. " numbered lists.
function formatMessage(text) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list);
    list = null;
  };
  text.split("\n").forEach((raw, i) => {
    const line = raw.trim();
    const bullet = line.match(/^[•\-]\s+(.*)/);
    const numbered = line.match(/^\d+\.\s+(.*)/);
    if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) {
        flush();
        list = { type, items: [] };
      }
      list.items.push(inline((bullet || numbered)[1], `li${i}`));
    } else {
      flush();
      if (line) blocks.push({ type: "p", content: inline(line, `p${i}`) });
    }
  });
  flush();
  return blocks.map((b, i) => {
    if (b.type === "p") return <p key={i}>{b.content}</p>;
    const List = b.type;
    return (
      <List key={i}>
        {b.items.map((item, j) => (
          <li key={j}>{item}</li>
        ))}
      </List>
    );
  });
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const q = text.trim();
    if (!q || loading) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });
      if (!res.ok) throw new Error("bad response");
      const data = await res.json();
      setMessages((m) => [
        ...m,
        { role: "bot", text: data.answer, source: data.source, method: data.method, confidence: data.confidence },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "bot", text: "Sorry, I can't reach the server right now. Please try again shortly." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="chat">
        <header>
          <h1>VIT Mumbai Enquiry Assistant</h1>
          <p>Vidyalankar Institute of Technology, Wadala</p>
        </header>

        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">
                {m.role === "bot" ? formatMessage(m.text) : m.text}
                {(m.method || (m.source && m.source.startsWith("http"))) && (
                  <div className="source">
                    {m.method && METHOD_LABEL[m.method] && (
                      <span className="method">
                        {METHOD_LABEL[m.method]}
                        {m.method !== "fallback" && ` · ${Math.round(m.confidence * 100)}%`}
                      </span>
                    )}
                    {m.source && m.source.startsWith("http") && (
                      <a href={m.source} target="_blank" rel="noreferrer">View official source</a>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="msg bot">
              <div className="bubble typing">Typing…</div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <div className="quick">
          {QUICK.map((q) => (
            <button key={q} onClick={() => send(q)} disabled={loading}>
              {q}
            </button>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your question…"
            maxLength={500}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
