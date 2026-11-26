# Day-4: Tools, Structured Output, and Google Sheets 🛠️📊

**Course:** AI Agents Workshop (Level 2: Single Agent)  
**Session:** Day 4 (Session 2 of 3)  
**Student:** Manindra Babu  
**Roll Number:** `24B11AI060`  
**Case Study:** Sweet Crumbs Home Bakery & Cafe  
**Model Provider:** Groq (`openai/gpt-oss-20b`) / Google Gemini (`gemini-3.5-flash`) via LangChain  
**Agent Framework:** LangChain Tool Calling (`langchain.tools` & `langchain.agents.create_agent`)  
**Production Temperature:** `0.2`

---

## 📌 Executive Summary
While Day 3 introduced the concept of an AI Agent and its ReAct loop, **Day 4** equips the agent with real capabilities to interact with the external world:
- **Why Agents Need Tools:** Pure LLMs hallucinate calculations, have stale training data, and cannot perform stateful external actions. Tools provide live facts, exact arithmetic, and external world effects.
- **Function Calling Mechanics:** The LLM does *not* execute code directly. It acts as the "wait staff" taking down an order ticket (generating JSON tool calls), while your Python code acts as the "kitchen" executing the function and handing back results (Tool Messages).
- **Writing Robust Tools:** Implementing `@tool` with clear typed signatures, comprehensive docstrings, and safe error handling.
- **Structured Output with Pydantic:** Using `pydantic.BaseModel` to enforce rigorous runtime data validation (`Literal` menu checks, quantity $\ge 1$) before saving records.
- **Persistent Storage Integration:** Seamlessly appending confirmed customer orders to Google Sheets via `gspread` with graceful fallback resilience for local and CI environments.
- **Execution Tracing:** Auditing multi-step tool call sequences and model thought processes.

---

## 🏗️ Architecture & Concepts Covered

### 1. The Three Reasons Agents Need Tools
```
+--------------------------------------------------------------------------+
|                        WHY AGENTS NEED TOOLS                             |
+--------------------+---------------------+-------------------------------+
|     Live Facts     |    Exact Maths      |         Real Actions          |
| Today's hours,     | Bill calculation,   | Writing to Google Sheets,     |
| real-time stock    | tax computations    | booking slots, sending emails |
+--------------------+---------------------+-------------------------------+
```

### 2. The Waiter & Kitchen Mental Model (How Tool Calling Works)
```
[User Message]
     │
     ▼
┌──────────────┐     1. Generates Tool Call Ticket      ┌─────────────────┐
│     LLM      │ ─────────────────────────────────────> │   Python Code   │
│ (The Waiter) │ <───────────────────────────────────── │  (The Kitchen)  │
└──────────────┘     2. Returns Tool Message Result     └─────────────────┘
     │                                                           │
     │ 3. Formulates Final Polite Response                       │ Calls external
     ▼                                                           ▼ APIs / Sheets
[Customer Confirmation]                                   [Google Sheets DB]
```
1. **User asks a question** requiring external data or action.
2. **Model chooses tool & writes ticket:** Instead of guessing, the LLM outputs a structured `tool_call` containing the function name and exact arguments.
3. **Python runtime executes tool:** Your application executes the native Python function and packages the return value into a `ToolMessage`.
4. **Model synthesizes answer:** The LLM receives the `ToolMessage` as ground truth and crafts a friendly, conversational reply.

---

## 📋 The Three Bakery Tools

Each tool is wrapped using LangChain's `@tool` decorator with explicit type hints and detailed docstrings:

```python
@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants (default is 1)
    """

@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. "sunday" or "monday"."""

@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Only call this after the customer confirms name, item, qty, and pickup date.

    Args:
        name: Customer first name
        item: "plum cake", "chocolate cake", or "cupcake"
        qty: Quantity to bake (must be >= 1)
        date: Requested pickup date (must be >= 24h notice)
        eggless: True if eggless is requested
    """
```

### Why Docstrings Matter
The LLM reads the tool's name, description, and argument documentation as its prompt instructions. If a tool has a vague name (e.g., `tool1`) or lacks argument descriptions, the model will misroute calls or pass invalid arguments.

---

## 🛡️ Structured Output & Pydantic Validation

To prevent corrupt or invalid data from reaching our Google Sheet database, we define a strict Pydantic schema:

```python
class Order(BaseModel):
    name: str = Field(description="Customer's first name", min_length=1)
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item name")
    qty: int = Field(description="Quantity ordered (must be at least 1)", ge=1)
    date: str = Field(description="Pickup date as provided by customer (e.g. '24 Dec', 'tomorrow')")
    eggless: bool = Field(description="True if customer requested eggless version, False otherwise")
```

### Guardrails Enforced:
1. **Allowed Items Only:** `Literal["plum cake", "chocolate cake", "cupcake"]` automatically rejects items like "donuts" or "croissants" at the schema level.
2. **Positive Quantities:** `Field(ge=1)` prevents zero or negative orders.
3. **Required Date & Name:** Prevents anonymous or dateless booking tickets.

---

## 📊 Google Sheets Order Persistence

The `SheetManager` class interfaces with Google Sheets using `gspread`:
- **Header Columns:** `['name', 'item', 'qty', 'date', 'eggless', 'total']`
- **Cloud Append:** Automatically calls `ws.append_row([name, item, qty, date, eggless, total])` when Google Cloud service credentials are authenticated.
- **Local Fallback:** In CLI / testing environments without OAuth tokens, orders are safely captured in memory so agent workflows never crash.

---

## 🧪 Verification Scenarios & Test Results

All scenarios and student experiments are integrated and verified directly in `Day-4/task.py`:

| Scenario ID | Test Objective | Customer Input | Tools Called | Outcome |
| :---: | :--- | :--- | :---: | :---: |
| **01** | Multi-Tool Coordination | *"Are you open on Sunday, and how much are 4 cupcakes?"* | `is_open`, `get_price` | **PASS [OK]** Quotes Sunday hours (10 am to 6 pm) and exact cupcake total (Rs 320). |
| **02** | Confirmed Order Logging | *"Hi, I'm Priya. Please book 2 plum cakes for tomorrow, eggless."* | `log_order` | **PASS [OK]** Validated via Pydantic, logged order to sheet (`Rs 1000`), confirmed booking. |
| **03** | Incomplete Order Guardrail | *"I want to order a chocolate cake."* | *None* | **PASS [OK]** Agent refuses to book without required fields; asks for pickup date and customer name. |
| **04** | Unsupported Menu Guardrail | *"Can I order 6 donuts for Friday?"* | `get_price` | **PASS [OK]** Tool catches unlisted item; agent politely explains donuts are not on the menu. |
| **05** | Structured Extraction | Direct `with_structured_output(Order)` on messy customer texts | *Extraction Engine* | **PASS [OK]** Successfully parsed name, item, qty, date, and eggless boolean. |
| **06** | Execution Trace Audit | Inspect full JSON tool calls and arguments | *Trace Inspector* | **PASS [OK]** Verified tool calls, argument mappings, and intermediate messages. |

---

## 💡 Student 30% Implementation Highlights

1. **Multi-Tool Orchestration:** Built multi-turn conversational loop maintaining conversation state and tracing tool call tickets.
2. **Pydantic Validation Guardrails:** Enforced strict schema validation inside `log_order` to prevent corrupt data insertion into Google Sheets.
3. **Local & Cloud Resilience:** Engineered `SheetManager` to automatically switch between `gspread` Google Cloud authentication and structured local registers without runtime crashes.
4. **Structured Output Parser:** Integrated and tested `llm.with_structured_output(Order)` for robust entity extraction from unstructured conversational prompts.
5. **Execution Trace Logging:** Implemented an execution trace utility that displays tool calls, argument payloads, and execution results for live debugging.
