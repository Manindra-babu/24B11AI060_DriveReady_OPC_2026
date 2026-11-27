# 24B11AI060 - DriveReady OPC 2026

**Student Name:** Manindra Babu  
**Roll Number:** 24B11AI060  
**Program:** DriveReady OPC 2026 · AI Agents Workshop (Level 2: Single Agent)  

---

## 📁 Repository Structure

```text
24B11AI060_DriveReady_OPC_2026/
├── Day-1/
│   ├── task.py
│   └── README.md
├── Day-2/
│   ├── task.py
│   └── README.md
├── Day-3/
│   ├── task.py
│   └── README.md
├── Day-4/
│   ├── task.py
│   └── README.md
└── Day-5/
    ├── task.py
    └── README.md
```

---

## 📅 Daily Progress Tracker

| Day | Topic | Status |
|:---:|:---|:---:|
| **Day-1** | How LLMs Work & First API Call (Sweet Crumbs Artisan Bakery) | ✅ Completed |
| **Day-2** | Prompts, Memory, and the Python Agents Need | ✅ Completed |
| **Day-3** | Your First Agent: What an Agent Is, and LangChain | ✅ Completed |
| **Day-4** | Tools, Structured Output, and Google Sheets | ✅ Completed |
| **Day-5** | Memory and RAG: An Agent That Remembers and Looks Things Up | ✅ Completed |

---

## 🧁 Module Summaries

### Day-1: How LLMs Work & First API Call
- **Focus:** Core LLM fundamentals (next-token prediction loop, tokens, temperature dial, context window, statelessness) and preventing hallucination using tailored **Facts and Rules**.
- **Model:** `openai/gpt-oss-120b` via Groq API (`langchain_groq`).
- **Prompt Safety:** Evaluated prompt injection vulnerabilities using `meta-llama/llama-prompt-guard-2-22m`.
- Detailed code and documentation: [`Day-1/`](./Day-1/)

### Day-2: Prompts, Memory, and the Python Agents Need
- **Focus:** Conversational memory management with sliding-window history trimming, few-shot intent classification, structured JSON order extraction, and core Python for agents (`@log_call` decorators, functions with type hints and docstrings, resilient `try / except` handling).
- Detailed code and documentation: [`Day-2/`](./Day-2/)

### Day-3: Your First Agent: What an Agent Is, and LangChain
- **Focus:** Distinguishing chatbots from agents, the four parts of an agent (brain, instructions, tools, memory), the ReAct agent loop, three levels of AI, architecture decision framework (plain code vs one LLM call vs agent), rebuilding with LangChain LCEL, temperature tuning (0.3 vs 0.5 vs 0.7), and automated evaluation with 100% pass rate.
- **Model:** Google Gemini (`gemini-3.5-flash`) via LangChain (`langchain-google-genai`).
- Detailed code and documentation: [`Day-3/`](./Day-3/)

### Day-4: Tools, Structured Output, and Google Sheets
- **Focus:** Equipping agents with external tools using LangChain's `@tool` decorator (`get_price`, `is_open`, `log_order`), function calling execution loop, Pydantic runtime data validation (`BaseModel`, `Literal`, `ge=1`), Google Sheets order persistence via `gspread` (with offline fallback resilience), and execution trace auditing.
- **Model:** Groq (`openai/gpt-oss-20b`) / Gemini (`gemini-3.5-flash`) via LangChain (`langchain.agents.create_agent`).
- Detailed code and documentation: [`Day-4/`](./Day-4/)

### Day-5: Memory and RAG — An Agent That Remembers and Looks Things Up
- **Focus:** Three kinds of memory (conversation, customer, knowledge base), document chunking with overlap, dense vector embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), vector database indexing (ChromaDB & normalized cosine inner-product index), retrieval-augmented generation (RAG) with strict anti-hallucination guardrails, custom testing story (*The Clockmaker of Silverfall* replacing *The Little Lantern*), and multi-turn thread memory isolation using LangGraph `InMemorySaver` with `thread_id`. Single agent Level 2 sign-off with 100% test pass rate.
- **Model:** Groq (`openai/gpt-oss-120b`) / Gemini (`gemini-3.5-flash`) via LangChain (`langchain.agents.create_agent`).
- Detailed code and documentation: [`Day-5/`](./Day-5/)
