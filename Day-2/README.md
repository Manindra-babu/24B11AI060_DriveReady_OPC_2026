# Day 2: Prompts, Memory, and the Python Agents Need

**Program:** DriveReady OPC 2026 · AI Agents Workshop (Level 1 Fundamentals)  
**Student Name:** Manindra Babu  
**Roll Number:** 24B11AI060  
**Business Case Study:** Sweet Crumbs Home Bakery  

---

## 📌 Core Concepts Learned in Day 2

### 1. Messages and Roles (`system`, `user`, `assistant`)
- A conversation sent to an LLM is structured as an ordered list of message dictionaries.
- **`system`**: Sets behavior, persona, facts, and boundaries (hidden from customer).
- **`user`**: Input query or statement submitted by the human.
- **`assistant`**: Model's previous responses. The model's primary objective is to generate the subsequent assistant turn.

### 2. The 4-Part System Prompt Architecture
A robust system prompt consists of four essential components:
- **ROLE:** Establishes identity and organizational context.
- **FACTS:** Injected domain knowledge (catalog, prices, lead times) to prevent hallucination.
- **RULES:** Explicit negative and positive operational constraints (e.g., never guarantee delivery arrival times).
- **FORMAT:** Output constraints (length, language, tone, structure).

### 3. Few-Shot Prompting (Demonstration vs Instruction)
- Providing 2 to 4 exemplar question-answer pairs directly in the messages history enforces exact output formatting (e.g., single-word category tags: `order`, `question`, `complaint`) that can be reliably consumed by downstream code.

### 4. Structured JSON Output Extraction
- While humans prefer conversational paragraphs, software systems require structured key-value attributes.
- By instructing the model to output strict JSON and parsing it with `json.loads` inside a `try/except` block, unstructured messages are parsed into machine-readable dictionaries.

### 5. Memory Management & Context Trimming
- LLMs are inherently stateless. Conversational persistence requires the application layer to maintain an append-only history list.
- To prevent context window saturation and escalating token costs, we implement **sliding-window trimming**:
  ```python
  recent_messages = [history[0]] + history[1:][-MAX_TURNS:]
  ```
  *(Preserving the system prompt while retaining only the most recent N conversational turns).*

### 6. Python for Agents: Dictionaries, Functions, Decorators & Error Handling
- **Dictionaries:** Fast key-value lookups for product catalogs (`MENU.get(item, default)` to prevent runtime `KeyError`).
- **Type Hints & Docstrings:** Providing explicit type signatures (`item: str, qty: int = 1 -> int`) and docstrings serves as the metadata schema when functions become agent tools.
- **Decorators (`@log_call`):** Wrapping functions to intercept arguments and log execution details transparently.
- **Robust Error Handling (`try / except`):** Intercepting `RateLimitError`, `APIConnectionError`, and `json.JSONDecodeError` to maintain resilience and provide user-friendly fallbacks.

---

## 🛠️ Code Implementation Walkthrough (`Day-2/task.py`)

### 1. Function with Type Hints & `@log_call` Decorator
```python
def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [Tool Log] {func.__name__}{args} -> {result!r}")
        return result
    return wrapper

@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the Sweet Crumbs menu."""
    return MENU[item.lower().strip()] * qty
```

### 2. Few-Shot Intent Classifier
```python
def classify_intent(message_text: str) -> str:
    messages = [
        {"role": "system", "content": "Classify into: order, question, or complaint. Return ONLY the single word."},
        {"role": "user", "content": "need 2 plum cakes for friday"},
        {"role": "assistant", "content": "order"},
        {"role": "user", "content": "do you have eggless versions?"},
        {"role": "assistant", "content": "question"},
        {"role": "user", "content": "the cake was dry yesterday"},
        {"role": "assistant", "content": "complaint"},
        {"role": "user", "content": message_text}
    ]
    ...
```

### 3. Structured JSON Order Extraction
```python
def extract_order_json(customer_message: str) -> dict:
    # Prompts model for strict JSON with keys: name, item, qty, date, eggless
    # Safely parses with json.loads and error handling
```

### 4. Multi-Turn Conversational Memory Class
```python
class BakeryChatbot:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        recent = [self.history[0]] + self.history[1:][-self.max_turns:]
        # Calls Groq model with recent context and error handling
        ...
```

---

## 🧪 Verified Execution Results

```text
=================================================================
DAY 2: PROMPTS, MEMORY & PYTHON TOOLS (SWEET CRUMBS BAKERY)
=================================================================

1. Function with Type Hints & @log_call Decorator:
   [Tool Log] get_price('chocolate cake', 2) -> 1300
   Calculated total: Rs 1300

2. Few-Shot Intent Classification:
   'Can I get 3 cupcakes tomorrow?' -> Category: [order]
   'Do you deliver to Madhapur?' -> Category: [question]
   'The bread was burnt and completely dry!' -> Category: [complaint]

3. Structured JSON Order Extraction:
   Input: 'Hi this is Ravi, I need 2 chocolate cakes for 28th Nov, please make them eggless.'
   Parsed JSON: {
     "name": "Ravi",
     "item": "chocolate cake",
     "qty": 2,
     "date": "28th Nov",
     "eggless": true
   }

4. Multi-Turn Conversational Memory:

User: Hi, my name is Ravi.
Bot:  Hi Ravi! 👋 Welcome to Sweet Crumbs Home Bakery. What would you like to order today? (We have plum cake, chocolate cake, cupcakes, and sourdough loaf.) Let me know any preferences—eggless, flavors, etc.

User: Can I get 2 chocolate cakes for tomorrow? Make them eggless.
Bot:  Sure thing, Ravi! I’ve noted 2 egg‑less chocolate cakes for tomorrow. We’ll have them ready for you after the 24‑hour prep window. Let me know a convenient time to pick them up. 🍰

User: What did I order so far, and what will be my total price?
Bot:  You’ve ordered 2 egg‑less chocolate cakes.  
Price: 2 × ₹650 = ₹1,300 total. Let me know when you’d like to pick them up!

=================================================================
Day 2 demonstrations completed successfully!
=================================================================
```

---

## 🚀 How to Run

```bash
python task.py
```
*(Ensure `GROQ_API_KEY` is configured in your environment or local `.env`)*
