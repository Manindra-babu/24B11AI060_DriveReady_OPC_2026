# Day-3: Your First Agent & LangChain 🤖

**Course:** AI Agents Workshop (Level 2: Single Agent)  
**Session:** Day 3 (Session 1 of 3)  
**Student:** Manindra Babu  
**Roll Number:** `24B11AI060`  
**Case Study:** Sweet Crumbs Home Bakery & Cafe  
**Model Provider:** Google Gemini (`gemini-3.5-flash` / `gemini-2.5-flash`) via LangChain (`langchain-google-genai`)

---

## 📌 Executive Summary
While Day 1 covered the raw API fundamentals and Day 2 established prompt engineering and conversational memory patterns, **Day 3** transitions from simple chatbots into **AI Agents**:
- **Chatbot vs. Agent:** A chatbot only *talks* (describing actions), whereas an agent is granted *tools* and autonomy to *take action* until a goal is achieved.
- **The Four Parts of an Agent:** Brain (LLM), Instructions (System Prompt), Tools (Python functions), and Memory (Conversation State).
- **The Agent Loop (ReAct):** $\text{Think} \rightarrow \text{Act} \rightarrow \text{Observe} \rightarrow \text{Repeat}$.
- **Framework Abstraction:** Rebuilding the Sweet Crumbs assistant using **LangChain Expression Language (LCEL)** to make models interchangeable like a universal travel adapter.
- **Automated Regression Testing (`tests.csv`):** Validating the assistant against customer queries with quantifiable pass rate metrics.

---

## 🏗️ Architecture & Concepts Covered

### 1. Chatbot vs. Agent
| Feature | Traditional Chatbot | AI Agent |
| :--- | :--- | :--- |
| **Output** | Words / Policy explanations | Goal execution + Verified status |
| **Capabilities** | Answers questions passively | Calls external tools & APIs |
| **Execution** | Single response generation | Multi-round iterative loop |
| **Bakery Example** | Tells user how to cancel an order | Looks up order, calls cancel API, verifies refund |

### 2. The Four Parts of an Agent
```
        +-----------------------------------------+
        |                  BRAIN                  |
        |        (Google Gemini 3.5 Flash)        |
        +--------------------+--------------------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
+-----------------+ +-----------------+ +-----------------+
|  INSTRUCTIONS   | |      TOOLS      | |     MEMORY      |
|  System Prompt  | | Python Functions| |  Chat History   |
| (Facts & Rules) | |(get_price, etc.)| |  (Multi-Turn)   |
+-----------------+ +-----------------+ +-----------------+
```

### 3. Three Levels of AI & Decision Framework
1. **Plain Code (Traditional Software):** Deterministic, fixed calculations with known steps (e.g., computing $500 \times 3 \times 1.05$ GST). Zero hallucinations, zero cost, instant latency.
2. **One LLM Call (Generative AI):** Single-shot text creation without tool execution (e.g., drafting an Instagram Diwali post or translating a greeting).
3. **Agentic AI:** Open-ended goals where steps cannot be known in advance and require external tools (e.g., parsing a complex customized order, looking up prices, verifying the 24h advance notice, checking delivery feasibility, and recording the booking).

---

## 🧪 Automated Testing & Evaluation (`tests.csv`)

To ensure prompt adjustments do not cause unintended regressions, all customer test questions are stored in `tests.csv` with mandatory validation keywords.

### Test Results Summary:
- **Total Tests Executed:** 5
- **Passed:** 5
- **Failed:** 0
- **Pass Rate:** **100.0%**

### Evaluation Log:
| Test ID | Question | Expected Keyword | Bot Actual Response | Result |
| :---: | :--- | :---: | :--- | :---: |
| **01** | `How much is a plum cake?` | `500` | "Hello! Our delicious Plum cake costs Rs 500. Please note that all our orders require at least 24 hours' notice..." | **PASS [OK]** |
| **02** | `Do you make eggless cakes?` | `eggless` | "Yes, absolutely! Every cake and cupcake at Sweet Crumbs has a 100% eggless version available upon request..." | **PASS [OK]** |
| **03** | `Can I get a cake today evening?` | `24 hours` | "Hello! I'm so sorry, but we cannot fulfill same-day orders. All our treats strictly require at least 24 hours' notice..." | **PASS [OK]** |
| **04** | `Do you deliver to Kakinada?` | `check with the owner` | "Hello! Thank you for reaching out to Sweet Crumbs Home Bakery. I'll check with the owner about delivery to Kakinada..." | **PASS [OK]** |
| **05** | `What is the price of 6 cupcakes?` | `480` | "Hello! Six cupcakes cost Rs 480 (at Rs 80 each). We also offer a 100% eggless version upon request..." | **PASS [OK]** |

---

## 💡 Student 30% "Do It Yourself" Implementation

1. **System Prompt Customization:**  
   Formulated a custom prompt strictly enforcing Sweet Crumbs' policies:
   - **Facts:** Plum cake (Rs 500), Chocolate cake (Rs 650), Cupcake (Rs 80 each), 100% eggless options.
   - **Rules:** Mandatory 24 hours' advance notice; location delivery queries require checking with the owner.
2. **Automated Regression Suite (`tests.csv`):**  
   Configured automated CSV test engine with keyword validation, achieving 100% pass rate.
3. **Temperature Sensitivity Experiment:**  
   Compared low temperature (`0.3`) for deterministic customer support vs high temperature (`1.0`). At $T=0.3$, adherence to bakery facts and rules is strictly consistent across repeated runs.
4. **Agent vs. Code Classification:**  
   Demonstrated architectural classification for billing (plain code), marketing posts (one LLM call), and complex multi-item booking (agent).
5. **Day-4 Tool Preparation:**  
   Authored typed helper functions `get_price(item: str, quantity: int = 1) -> int` and `is_available(item: str, date_str: str) -> bool` with docstrings in preparation for tool calling in Day 4.

---

## 🔐 Security & Credential Hygiene
- **Zero API Keys in Repository:** API keys are never hardcoded. All calls load dynamically from `.env` via `python-dotenv`.
- **Git Protection:** `.env` is isolated locally in `.git/info/exclude`.
