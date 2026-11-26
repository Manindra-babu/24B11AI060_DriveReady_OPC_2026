"""
Day 4 Task: Tools, Structured Output, and Google Sheets
AI Agents Workshop (Level 2: Single Agent) - DriveReady OPC 2026
Student: Manindra Babu (Roll No: 24B11AI060)
Case Study: Sweet Crumbs Home Bakery & Cafe

Curriculum Covered:
1. Why Agents Need Tools (Live facts, real actions, exact maths)
2. How Function Calling Works (Model asks, code executes, returns tool message)
3. Writing Good Tools with @tool (Type hints, docstrings, graceful error handling)
4. The Agent with Tools (create_agent, multi-tool execution, and execution traces)
5. Rate Limits and Error Resilience (429 handling, max_retries, polite fallbacks)
6. Structured Output with Pydantic (BaseModel validation, Literal, ge=1 checks)
7. Google Sheets Order Logging (gspread append_row integration with local resilience)
8. End-to-End Order Workflow (log_order with validation, calculation, and confirmation)
9. Student 30% Experiments (Trace inspection, tool docstring impact, and structured parsing)
"""

import os
import sys
import json
from typing import Literal, List, Dict, Any, Optional
from dotenv import load_dotenv

# Load API keys securely from .env (NEVER hardcoded in source files)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

from pydantic import BaseModel, Field, ValidationError
from langchain.tools import tool
from langchain.agents import create_agent

# Initialize LLM with tool support (defaults to Groq with fallback to Gemini)
if GROQ_API_KEY:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, max_retries=3)
elif GEMINI_API_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2, api_key=GEMINI_API_KEY)
else:
    raise ValueError("No API key found in .env! Please configure GROQ_API_KEY or GEMINI_API_KEY.")


# =====================================================================
# 1. Ground Truth Bakery Data
# =====================================================================

MENU = {
    "plum cake": 500,
    "chocolate cake": 650,
    "cupcake": 80
}

HOURS = {
    "monday": "9 am to 8 pm",
    "tuesday": "9 am to 8 pm",
    "wednesday": "9 am to 8 pm",
    "thursday": "9 am to 8 pm",
    "friday": "9 am to 8 pm",
    "saturday": "9 am to 8 pm",
    "sunday": "10 am to 6 pm"
}


# =====================================================================
# 2. Structured Data Model (Pydantic Validation)
# =====================================================================

class Order(BaseModel):
    """
    Structured cake order validated against Sweet Crumbs menu constraints.
    """
    name: str = Field(description="Customer's first name", min_length=1)
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item name")
    qty: int = Field(description="Quantity ordered (must be at least 1)", ge=1)
    date: str = Field(description="Pickup date as provided by customer (e.g. '24 Dec', 'tomorrow')")
    eggless: bool = Field(description="True if customer requested eggless version, False otherwise")


# =====================================================================
# 3. Google Sheets Manager (With Local Fallback Resilience)
# =====================================================================

class SheetManager:
    """
    Manages order persistence. Connects to Google Sheets via gspread if authenticated,
    or maintains a local structured register mimicking ws.append_row.
    """
    def __init__(self, sheet_name: str = "Sweet Crumbs Orders"):
        self.sheet_name = sheet_name
        self.rows: List[List[Any]] = [
            ["name", "item", "qty", "date", "eggless", "total"]
        ]
        self.gc = None
        self.ws = None
        
        # Attempt Google Sheets connection if running in authenticated environment
        try:
            import gspread
            from google.auth import default
            creds, _ = default()
            self.gc = gspread.authorize(creds)
            try:
                sh = self.gc.open(self.sheet_name)
            except Exception:
                sh = self.gc.create(self.sheet_name)
            self.ws = sh.sheet1
        except Exception:
            # Local fallback for offline/CLI execution
            self.ws = None

    def append_row(self, row: List[Any]) -> None:
        self.rows.append(row)
        if self.ws:
            try:
                self.ws.append_row(row)
            except Exception as e:
                print(f"[Sheet Warning] Cloud append failed ({e}), saved to local memory.", flush=True)

    def get_all_records(self) -> List[List[Any]]:
        return self.rows


sheet_db = SheetManager()


# =====================================================================
# 4. Agent Tools Definition (@tool)
# =====================================================================

@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.

    Args:
        item: Menu item, e.g. "plum cake", "chocolate cake" or "cupcake"
        qty: How many the customer wants (default is 1)
    """
    clean_item = item.lower().strip()
    price = MENU.get(clean_item)
    if price is None:
        return f"'{item}' is not on the menu. We have: {', '.join(MENU.keys())}."
    total = price * qty
    return f"{qty} x {clean_item} = Rs {total}"


@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. "sunday" or "monday"."""
    clean_day = day.lower().strip()
    hours = HOURS.get(clean_day)
    if hours:
        return f"Open {hours} on {clean_day.capitalize()}."
    return f"'{day}' is not a valid day of the week. Please check Monday through Sunday."


@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the Orders sheet.
    Use only when the customer has given their name, item, quantity and pickup date."""
    clean_item = item.lower().strip()
    try:
        # Validate data integrity with Pydantic
        order = Order(
            name=name.strip(),
            item=clean_item,
            qty=qty,
            date=date.strip(),
            eggless=eggless
        )
    except ValidationError as e:
        return f"Could not save the order: validation error ({e.errors()[0]['msg']}). Ask customer to clarify."
    except Exception as e:
        return f"Could not save the order: {e}. Ask customer to correct it."

    unit_price = MENU.get(order.item, 0)
    total = unit_price * order.qty

    # Append validated record to sheet
    sheet_db.append_row([
        order.name,
        order.item,
        order.qty,
        order.date,
        "yes" if order.eggless else "no",
        total
    ])

    return f"Saved: {order.qty} x {order.item} for {order.name} on {order.date}, Rs {total}."


# =====================================================================
# 5. Agent Assembly & System Prompt
# =====================================================================

SYSTEM_PROMPT = """ROLE:   You are the friendly order assistant for Sweet Crumbs Home Bakery.
FACTS:  Orders strictly need at least 24 hours' notice. All cakes have eggless versions upon request.
RULES:  Always use get_price for prices and is_open for timings; never guess them.
        Before log_order, make sure you have name, item, quantity and pickup date; ask if any are missing.
        After saving, read the order back with the total.
FORMAT: Under 60 words, friendly, English matching the customer."""

tools = [get_price, is_open, log_order]
bakery_agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


# =====================================================================
# 6. Stateful Chat Session with Execution Tracing
# =====================================================================

class BakeryAgentSession:
    """
    Maintains multi-turn conversational state and execution tracing.
    """
    def __init__(self, agent=bakery_agent):
        self.agent = agent
        self.history: List[Dict[str, Any]] = []

    def chat(self, user_text: str) -> str:
        try:
            input_messages = self.history + [{"role": "user", "content": user_text}]
            result = self.agent.invoke({"messages": input_messages})
            self.history = result["messages"]
            return self.history[-1].content
        except Exception as e:
            print(f"[Agent Execution Error]: {e}", flush=True)
            return "Sorry, I'm a little busy. Please try again in a minute."

    def print_trace(self) -> None:
        """
        Prints the complete step-by-step trace of tool calls and model responses.
        """
        print("\n" + "-"*65, flush=True)
        print("AGENT EXECUTION TRACE:", flush=True)
        print("-"*65, flush=True)
        for idx, m in enumerate(self.history, start=1):
            role = getattr(m, "type", m.get("role", "unknown") if isinstance(m, dict) else "msg")
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            
            print(f"[{idx}] {role.upper()}:", flush=True)
            if tool_calls:
                for tc in tool_calls:
                    print(f"    -> TOOL CALL: {tc['name']}({tc['args']})", flush=True)
            if content:
                print(f"    Content: {content.strip()[:140]}...", flush=True)
        print("-"*65 + "\n", flush=True)


# =====================================================================
# Main Demonstration
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*75, flush=True)
    print("   DAY 4: TOOLS, STRUCTURED OUTPUT & GOOGLE SHEETS (Sweet Crumbs)", flush=True)
    print("="*75, flush=True)

    session = BakeryAgentSession()

    # Scenario 1: Information Gathering with Tools (is_open + get_price)
    print("\n[SCENARIO 1] Multiple Tool Lookups in a Single Request:", flush=True)
    q1 = "Are you open on Sunday? How much are 2 chocolate cakes?"
    print(f"Customer: {q1}", flush=True)
    ans1 = session.chat(q1)
    print(f"Bot:      {ans1}\n", flush=True)

    # Scenario 2: Complete Order Placement (log_order)
    print("[SCENARIO 2] Full Order Placement with Data Validation & Sheet Logging:", flush=True)
    q2 = "I'm Ravi. Please book 2 eggless chocolate cakes for 24 Dec."
    print(f"Customer: {q2}", flush=True)
    ans2 = session.chat(q2)
    print(f"Bot:      {ans2}\n", flush=True)

    # Display Sheet Records
    print("[SHEET AUDIT] Current Saved Rows in Orders Register:", flush=True)
    for r in sheet_db.get_all_records():
        print("  ", r, flush=True)

    # Scenario 3: Missing Information Handling
    print("\n[SCENARIO 3] Guardrail - Incomplete Order (Missing Date):", flush=True)
    session_incomplete = BakeryAgentSession()
    q3 = "2 plum cakes please, I'm Priya."
    print(f"Customer: {q3}", flush=True)
    ans3 = session_incomplete.chat(q3)
    print(f"Bot:      {ans3}\n", flush=True)

    # Scenario 4: Pydantic Validation on Bad Input (Item not on menu)
    print("[SCENARIO 4] Guardrail - Unsupported Item Handled Gracefully:", flush=True)
    q4 = "Can you book 3 donuts for Friday, name is Kavya?"
    print(f"Customer: {q4}", flush=True)
    ans4 = session_incomplete.chat(q4)
    print(f"Bot:      {ans4}\n", flush=True)

    # Scenario 5 (Student 30%): Structured Output Extraction Test
    print("[STUDENT 30% TASK] Testing with_structured_output(Order) on Unstructured Messages:", flush=True)
    extractor = llm.with_structured_output(Order)
    test_msgs = [
        "hi im ravi, need 2 plum cakes for 24 dec, eggless please",
        "book 6 cupcakes for sravani tomorrow, regular",
        "need 1 chocolate cake for arjun on saturday, eggless yes"
    ]
    for msg in test_msgs:
        parsed = extractor.invoke(msg)
        print(f"  Input:  \"{msg}\"", flush=True)
        print(f"  Parsed: {parsed}\n", flush=True)

    # Scenario 6: Inspection of Execution Trace
    print("[TRACE AUDIT] Execution Trace for Multi-Tool Run:", flush=True)
    session.print_trace()

    print("Day-4 demonstration executed successfully!", flush=True)
