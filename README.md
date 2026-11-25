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
│   ├── README.md
│   └── agenticAI_Day1.ipynb
├── Day-2/
│   ├── task.py
│   └── README.md
└── Day-3/
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
