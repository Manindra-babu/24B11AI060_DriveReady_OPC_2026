# Day-5: Memory and RAG — An Agent That Remembers and Looks Things Up 🧠🔍

**Course:** AI Agents Workshop (Level 2: Single Agent)  
**Session:** Day 5 (Session 3 of 3)  
**Student:** Manindra Babu  
**Roll Number:** `24B11AI060`  
**Knowledge Story Case Study:** *The Clockmaker of Silverfall* (Custom Original Story replacing "The Little Lantern")  
**Production Agent Case Study:** *Sweet Crumbs Home Bakery & Cafe*  
**Model Provider:** Groq (`openai/gpt-oss-120b`) / Google Gemini (`gemini-3.5-flash`) via LangChain  
**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)  
**Vector Databases:** ChromaDB (`sweet_crumbs_faq`) + Normalized In-Memory Cosine Index (FAISS-compatible `IndexFlatIP`)  
**Memory Architecture:** LangGraph Checkpointer (`InMemorySaver`) with `thread_id` conversation isolation  

---

## 📌 Executive Summary
While Day 4 equipped our assistant with external Python tools (`get_price`, `is_open`, `log_order`), **Day 5** completes the single agent architecture by solving two fundamental challenges:
1. **Conversation Memory (Short-Term):** Isolating multi-turn dialogues across different concurrent users using a checkpointer (`InMemorySaver`) and distinct `thread_id` identifiers.
2. **Knowledge Retrieval (RAG):** Searching extensive external documents and business policies dynamically on demand using semantic vector search instead of cramming hundreds of rules into the static prompt.
3. **Custom Testing Knowledge Story:** In accordance with the project requirements, we replaced the short "The Little Lantern" PDF with a rich, testable original narrative titled **"The Clockmaker of Silverfall"**, featuring specific character names, dates, artifacts, locations, mechanical details, and quotes to rigorously validate factual retrieval, semantic inference, and zero-hallucination guardrails.
4. **Single-Agent Final Sign-Off:** Bringing together all 5 components of a Level 2 Agent:
   - **Brain:** LLM (`openai/gpt-oss-120b`)
   - **Instructions:** System Prompt following Role-Facts-Rules-Format (RFRF)
   - **Tools:** `get_price`, `is_open`, `log_order`, and `search_faq`
   - **Memory:** Multi-turn isolated session threads (`thread_id`)
   - **Knowledge:** Vector databases (ChromaDB & FAISS-style normalized index)

---

## 🏗️ Architecture & Core Concepts Covered

### 1. Three Kinds of Agent Memory
```
+-----------------------------------------------------------------------------------------+
|                                  THREE KINDS OF MEMORY                                  |
+------------------------------+------------------------------+---------------------------+
| 1. Conversation Memory       | 2. Customer Memory           | 3. Knowledge Base (RAG)   |
| (Short-Term)                 | (Long-Term)                  | (Document Memory)         |
| Active chat history          | Saved across sessions        | Business policies & FAQs  |
| "Ravi wants 2 plum cakes"    | "Ravi prefers eggless"       | "Free delivery > Rs 1,000"|
| Scope: One conversation      | Scope: Weeks / Months        | Scope: Until docs update  |
| Implemented via: thread_id   | Implemented via: Database    | Implemented via: Vector DB|
+------------------------------+------------------------------+---------------------------+
```

### 2. Why Not Put Everything in the System Prompt?
| Limitation | All Facts in System Prompt | Retrieval-Augmented Generation (RAG) |
| :--- | :--- | :--- |
| **Input Cost** | High (charged for hundreds of facts on *every* call) | **Low** (sends only top 2–3 relevant chunks) |
| **Latency** | Slower (model processes large static prompt) | **Fast** (compact prompt, fast vector search) |
| **Attention Dilution** | Important facts get lost among irrelevant rules | **Focused** (model attends only to relevant context) |
| **Knowledge Updates** | Requires rewriting and testing prompt templates | **Instant** (simply add or update document chunks) |

---

## 📖 The Custom Knowledge Story: "The Clockmaker of Silverfall"

Instead of using "The Little Lantern", we authored a multi-dimensional narrative with specific, verifiable facts:

> **Narrative Synopsis:**  
> In the mist-shrouded valley of Silverfall beneath the Azure Ridge, Master Silas spent forty years constructing the **Celestial Chronometer**—a brass timepiece featuring 36 synthetic sapphire bearings, four counter-weighted silver escapements, and a pendulum sculpted from starlight obsidian. The timepiece required winding every winter solstice using the **Solstice Key**, a golden meteorite brass key kept in a cedar lockbox in the attic.
> 
> During the **Great Frost of November 1782**, freezing winds cracked the Cathedral bell gears, disabling the valley's only timekeeping and relief coordination. North at Raven's Pass, an eight-wagon caravan led by Captain Orin broke an axle on jagged rocks. Silas used the Chronometer's astrolabe dial to compute the lunar azimuth and activated the clocktower's nocturnal beacon—an oil-furnace spotlight focused through a 60mm quartz prism. The amber light guided 18 freezing travelers to safety. When offered 200 gold ducats, Silas refused reward, stating: *"True precision is never measured by how we count our hours, but by how each hour serves someone who has lost their way in the storm."*

---

## ⚙️ Technical Implementation: RAG Pipeline

### 1. Semantic Chunking with Sliding Overlap
```python
def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    # Splits document into 400-char segments with 80-char overlap
    # Prevents cutting sentences or key numerical entities in half
```

### 2. Embeddings via SentenceTransformers
Text strings are projected into a 384-dimensional dense vector space using `all-MiniLM-L6-v2`. Semantic similarity is computed as the inner product of normalized vectors:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q}_{\text{norm}} \cdot \mathbf{d}_{\text{norm}}$$

### 3. Vector Database Indexing (FAISS & ChromaDB)
- **Custom Story:** Indexed into an exact normalized inner-product vector index (`VectorIndex`) implementing FAISS-compatible $O(1)$ batch vector search.
- **Bakery Policies:** Indexed into **ChromaDB** (`sweet_crumbs_faq`) with 12 distinct operational policies (delivery tiers, advance payment, cancellations, allergen advisories, and photo cake notice).

### 4. Grounded Prompt with Anti-Hallucination Guardrail
```text
SYSTEM: You are an evidence-grounded question-answering assistant.
Answer the question ONLY using the provided text context.
If the answer cannot be deduced strictly from the context, respond exactly:
'The answer is not available in the context.'
Do not fabricate or extrapolate information. Keep answers factual and concise.
```

---

## 🧵 Multi-Turn Thread Memory (LangGraph `InMemorySaver`)

To ensure customers chatting simultaneously never see or interfere with each other's data:
1. The agent is initialized with `checkpointer=InMemorySaver()`.
2. Each customer interaction passes a unique `config={"configurable": {"thread_id": "..."}}`.
3. Customer **Ravi** (`thread_id="ravi"`) asks about Sunday hours and Rs 1,200 delivery.
4. Customer **Sravani** (`thread_id="sravani"`) asks about plum cake nut allergens and cancellation.
5. In Turn 5, when Ravi asks: *"What is my name and will I pay for delivery?"*, the agent correctly retrieves Ravi's name and confirms free delivery without confusing Sravani's concurrent questions.

---

## 🧪 Automated Evaluation & Verification Suite

All tests are integrated into `Day-5/task.py` and execute automatically:

| Test ID | Category | Query / Prompt | Expected Keywords | Bot Actual Output | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **STORY-01** | Factual Retrieval | *What is the Celestial Chronometer made of, and who built it?* | `Silas`, `sapphire`, `obsidian` | "The Celestial Chronometer is a brass timepiece equipped with thirty-six synthetic sapphire bearings, four counter-weighted silver escapements, and a central pendulum of starlight obsidian. It was built by Silas." | **PASS [OK]** |
| **STORY-02** | Specific Key & Location | *What key winds the clock and where is it kept?* | `Solstice Key`, `cedar lockbox` | "The clock is wound with the Solstice Key, and it is kept in a velvet-lined cedar lockbox on the workshop's second floor." | **PASS [OK]** |
| **STORY-03** | Date & Tempest Event | *What happened during the Great Frost of November 1782?* | `Cathedral`, `bell`, `tempest` | "During the Great Frost of November 1782 a sudden polar tempest swept through the mountain gaps, dropping temperatures far below freezing and covering Silverfall with four feet of snow. The intense freezing winds cracked the iron gears of the Cathedral bell..." | **PASS [OK]** |
| **STORY-04** | Rescue Mechanism | *How did Silas signal the stranded travelers at Raven's Pass?* | `beacon`, `amber`, `prism` | "He activated the clocktower's nocturnal beacon—a high-intensity spotlight powered by an oil furnace and focused through a sixty-millimeter quartz prism—sending a bright amber beam that guided the stranded travelers." | **PASS [OK]** |
| **STORY-05** | Philosophical Quote | *What was Silas's response when offered two hundred gold ducats?* | `precision`, `hours`, `lost` | "He gently returned the coin, saying, 'True precision is never measured by how we count our hours, but by how each hour serves someone who has lost their way in the storm.'" | **PASS [OK]** |
| **STORY-06** | Anti-Hallucination Guardrail | *Did Master Silas build a steam-powered flying airship to travel to Mars?* | `not available in the context` | "The answer is not available in the context." | **PASS [OK]** |
| **THREAD-01** | Thread Memory Recall | *What is my name and will I pay for delivery?* (Customer Ravi, Turn 5) | `Ravi`, `free` | "Your name is Ravi, and you won’t be charged for delivery since your ₹1,200 order qualifies for free delivery. Let me know the address and time you’d like it delivered!" | **PASS [OK]** |

### 📈 Test Score Summary:
- **Total Tests Executed:** 7
- **Passed:** 7
- **Failed:** 0
- **Final Pass Rate:** **100.0%**

---

## 💡 Student 30% Implementation Highlights

1. **Custom Narrative Authoring:**  
   Replaced "The Little Lantern" with "The Clockmaker of Silverfall", providing precise semantic anchors for testing.
2. **Normalized Vector Index Architecture:**  
   Implemented an in-memory normalized vector index calculating cosine similarity through matrix dot products, ensuring zero-dependency compatibility with FAISS.
3. **Zero-Hallucination Prompt Engineering:**  
   Engineered strict out-of-context guardrails so questions outside the knowledge base reliably return `"The answer is not available in the context."` without fabricating facts.
4. **Thread-Level Conversation Isolation:**  
   Demonstrated multi-turn conversational persistence with LangGraph checkpointers, verifying concurrent user separation across independent sessions.
5. **RAG-Powered Policy Agent:**  
   Integrated ChromaDB collection querying inside the `@tool def search_faq` function, allowing the Sweet Crumbs agent to dynamically answer policy questions.
