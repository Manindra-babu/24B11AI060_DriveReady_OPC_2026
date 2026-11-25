"""
Day 3 Task: Your First Agent - What an Agent Is, and LangChain
AI Agents Workshop (Level 2: Single Agent) - DriveReady OPC 2026
Student: Manindra Babu (Roll No: 24B11AI060)
Case Study: Sweet Crumbs Home Bakery & Cafe

Curriculum Covered:
1. Chatbot vs Agent (Talking vs Doing)
2. Four Parts of an Agent (Brain, Instructions, Tools, Memory)
3. The Agent Loop (Think -> Act -> Observe -> Repeat)
4. Three Levels of AI (Traditional, Generative, Agentic)
5. Agent or Not Decision Framework (Plain code vs One LLM call vs Agent)
6. Why LangChain (Unified interface, interchangeable model adapters)
7. LangChain Basics (ChatPromptTemplate, ChatGoogleGenerativeAI, LCEL Chain)
8. Automated Evaluation & Regression Testing (tests.csv with PASS/FAIL metrics)
9. Student 30% Tasks (Custom prompt, tests.csv evaluation, temperature testing, Day-4 tool preparation)
"""

import os
import csv
import time
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load API keys from .env file (NEVER hardcoded in source code)
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# =====================================================================
# 1. System Prompt Definition (Role, Facts, Rules, Format)
# =====================================================================

SYSTEM_PROMPT = """ROLE: You are the friendly, polite assistant for Sweet Crumbs Home Bakery.
FACTS:
- Plum cake: Rs 500
- Chocolate cake: Rs 650
- Cupcake: Rs 80 each (e.g. 6 cupcakes = Rs 480)
- Notice requirement: All orders strictly need at least 24 hours' notice. We cannot do same-day orders.
- Dietary options: Every cake and cupcake has a 100% eggless version available upon request.
RULES:
- Never promise immediate or same-day delivery; always state that orders require 24 hours' notice.
- If a customer asks about delivery to specific cities/locations (like Kakinada or outside our standard area), or anything not in FACTS, explicitly state: "I'll check with the owner."
- For multi-item or quantity calculations, work out the total price accurately.
FORMAT: Under 50 words, friendly, polite English."""


# =====================================================================
# 2. Assistant Factory using LangChain LCEL
# =====================================================================

def build_assistant(system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.3):
    """
    Builds a LangChain LCEL pipeline:
    PromptTemplate -> ChatGoogleGenerativeAI (Gemini)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=temperature,
        api_key=GEMINI_KEY
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    return chain


def safe_extract_text(ai_message) -> str:
    """
    Extracts plain text response from LangChain AIMessage content,
    handling both string format and structured content blocks.
    """
    content = ai_message.content if hasattr(ai_message, "content") else ai_message
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"].strip()
        return str(first).strip()
    return str(content).strip()


# =====================================================================
# 3. Conversational Memory Demonstration
# =====================================================================

class SweetCrumbsAssistant:
    """
    Assistant with conversational memory tracking using LangChain.
    """
    def __init__(self, system_prompt: str = SYSTEM_PROMPT, temperature: float = 0.3):
        self.chain = build_assistant(system_prompt, temperature)
        self.history: List[Tuple[str, str]] = []

    def ask(self, question: str) -> str:
        res = self.chain.invoke({
            "history": self.history,
            "question": question
        })
        answer = safe_extract_text(res)
        self.history.extend([("human", question), ("ai", answer)])
        return answer

    def reset_memory(self) -> None:
        self.history = []


# =====================================================================
# 4. Automated Evaluation Engine (tests.csv)
# =====================================================================

def run_evaluation(test_csv_path: str = "Day-3/tests.csv", temperature: float = 0.3) -> Tuple[int, int, float]:
    """
    Executes automated regression tests from tests.csv.
    Evaluates whether the assistant's output contains the mandatory keyword.
    """
    if not os.path.exists(test_csv_path):
        test_csv_path = "tests.csv"
        
    chain = build_assistant(SYSTEM_PROMPT, temperature=temperature)
    rows = list(csv.DictReader(open(test_csv_path, "r", encoding="utf-8")))
    
    passed = 0
    total = len(rows)
    print(f"\n{'='*75}", flush=True)
    print(f"RUNNING AUTOMATED EVALUATION ON {total} TEST QUERIES (Temp: {temperature})", flush=True)
    print(f"{'='*75}", flush=True)
    
    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        must_include = row["must_include"].strip()
        
        res = chain.invoke({"history": [], "question": question})
        answer = safe_extract_text(res)
        time.sleep(1.0)
        
        is_pass = must_include.lower() in answer.lower()
        if is_pass:
            passed += 1
            status_tag = "PASS [OK]"
        else:
            status_tag = "FAIL [X]"
            
        print(f"Test {idx:02d}: {status_tag} | Q: \"{question}\"", flush=True)
        print(f"         Expected Keyword: '{must_include}'", flush=True)
        print(f"         Bot Answer:       \"{answer}\"", flush=True)
        print(f"{'-'*75}", flush=True)
        
    pass_rate = (passed / total) * 100
    print(f"\nFinal Test Score: {passed}/{total} Passed ({pass_rate:.1f}%)\n", flush=True)
    return passed, total, pass_rate


# =====================================================================
# 5. Agent or Not? Architecture Classification
# =====================================================================

def classify_architecture_tasks() -> List[Dict[str, str]]:
    """
    Classification of common tasks in the bakery into:
    - Level 1: Plain Code (Deterministic rules, fixed math)
    - Level 2: One LLM Call (Content generation, translation)
    - Level 3: Agent (Multi-step planning, tool invocation, verification)
    """
    tasks = [
        {
            "task": "Calculate exact total bill with 5% GST for 3 plum cakes",
            "decision": "Plain Code",
            "reason": "Deterministic calculation (3 * 500 * 1.05). Code is faster, free of cost, and never makes arithmetic mistakes."
        },
        {
            "task": "Write an engaging Diwali promotional greeting for Instagram",
            "decision": "One LLM Call",
            "reason": "Single creative text generation without requiring external tool execution."
        },
        {
            "task": "Handle 'I need 2 plum cakes + 6 cupcakes for Saturday, eggless, can you deliver to Kakinada?'",
            "decision": "Agent",
            "reason": "Requires parsing multiple items, looking up prices, verifying the 24h advance notice, checking delivery feasibility with the owner, and saving the order."
        }
    ]
    return tasks


# =====================================================================
# 6. Preview: Tool Calling Preparation for Day 4
# =====================================================================

def get_price(item: str, quantity: int = 1) -> int:
    """
    Look up the total price for bakery menu items.
    
    Args:
        item: The name of the pastry or cake.
        quantity: The quantity requested (default: 1).
        
    Returns:
        Total price in INR (Rs).
    """
    prices = {
        "plum cake": 500,
        "chocolate cake": 650,
        "cupcake": 80
    }
    item_clean = item.lower().strip()
    unit_price = prices.get(item_clean, 0)
    return unit_price * quantity


def is_available(item: str, date_str: str) -> bool:
    """
    Check if a bakery item is available for a given date.
    
    Args:
        item: The requested item name.
        date_str: Date for requested delivery/pickup.
        
    Returns:
        True if item is available on the specified date, False otherwise.
    """
    catalog = ["plum cake", "chocolate cake", "cupcake"]
    return item.lower().strip() in catalog


# =====================================================================
# Main Execution Demo
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*75, flush=True)
    print("   DAY 3: YOUR FIRST AGENT & LANGCHAIN (Sweet Crumbs Bakery)", flush=True)
    print("="*75, flush=True)

    # 1. Multi-turn Conversational Memory Demo
    print("\n[PART 1] Multi-turn Conversational Memory Test:", flush=True)
    bot = SweetCrumbsAssistant(temperature=0.3)
    turn1 = bot.ask("Hi, I'm Ravi. Do you make eggless cakes?")
    print(f"Customer: Hi, I'm Ravi. Do you make eggless cakes?\nBot:      {turn1}\n", flush=True)
    
    time.sleep(1.0)
    turn2 = bot.ask("What's my name, and can I get a chocolate cake today evening?")
    print(f"Customer: What's my name, and can I get a chocolate cake today evening?\nBot:      {turn2}\n", flush=True)

    # 2. Automated Evaluation on Customer Questions (tests.csv)
    print("\n[PART 2] Automated Regression Testing (tests.csv Evaluation):", flush=True)
    passed, total, rate = run_evaluation("Day-3/tests.csv", temperature=0.3)

    # 3. Architecture Decision Demonstration
    print("\n[PART 3] Architecture Decision: Plain Code vs LLM vs Agent:", flush=True)
    for task_info in classify_architecture_tasks():
        print(f"- Task: {task_info['task']}", flush=True)
        print(f"  Decision: {task_info['decision']}", flush=True)
        print(f"  Rationale: {task_info['reason']}\n", flush=True)

    # 4. Tool Preparation for Day 4
    print("[PART 4] Day-4 Tools Preview:", flush=True)
    print(f"get_price('plum cake', 3) = Rs {get_price('plum cake', 3)}", flush=True)
    print(f"get_price('cupcake', 6)   = Rs {get_price('cupcake', 6)}", flush=True)
    print(f"is_available('chocolate cake', 'tomorrow') = {is_available('chocolate cake', 'tomorrow')}", flush=True)
    print("\nDay-3 demonstration executed successfully!", flush=True)
