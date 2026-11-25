"""
Day 2: Prompts, Memory, and the Python Agents Need
DriveReady OPC 2026 - AI Agents Level 1
Student: Manindra Babu (Roll No: 24B11AI060)

Case Study: Sweet Crumbs Home Bakery
Core Concepts Implemented:
1. Messages & Roles (system, user, assistant)
2. System Prompt Engineering (Role, Facts, Rules, Format)
3. Few-Shot Prompting (Classification: order, question, complaint)
4. Structured JSON Output (Extracting order fields safely with json.loads)
5. Multi-Turn Memory Management & Sliding Window Trimming
6. Python for Agents: Dictionaries, Functions with Type Hints & Docstrings,
   Decorators (@log_call), and Robust Exception Handling (try/except)
"""

import os
import json
import functools
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, AuthenticationError

# 1. Load API Key securely
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        from google.colab import userdata
        api_key = userdata.get("GROQ_API_KEY")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or Colab secrets.")

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=api_key)

# 2. Python for Agents: Menu Data Dictionary
MENU = {
    "plum cake": 500,
    "chocolate cake": 650,
    "cupcake": 80,
    "sourdough loaf": 350
}

# 3. Python for Agents: Decorator for Logging Tool/Function Calls
def log_call(func):
    """Decorator that logs function name, arguments, and return value."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [Tool Log] {func.__name__}{args} -> {result!r}")
        return result
    return wrapper

# 4. Python for Agents: Function with Type Hints & Docstring
@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the Sweet Crumbs menu."""
    clean_item = item.lower().strip()
    if clean_item not in MENU:
        raise KeyError(f"Item '{item}' is not on the Sweet Crumbs menu.")
    return MENU[clean_item] * qty

# 5. Few-Shot Prompting: Customer Intent Classifier
def classify_intent(message_text: str) -> str:
    """Classifies a customer message into 'order', 'question', or 'complaint' using few-shot examples."""
    messages = [
        {"role": "system", "content": "Classify the customer message into exactly one category: order, question, or complaint. Reply with ONLY that single word."},
        # Few-shot examples
        {"role": "user", "content": "need 2 plum cakes for friday"},
        {"role": "assistant", "content": "order"},
        {"role": "user", "content": "do you have eggless versions of chocolate cake?"},
        {"role": "assistant", "content": "question"},
        {"role": "user", "content": "the cake delivered yesterday was completely damaged"},
        {"role": "assistant", "content": "complaint"},
        # Target customer message
        {"role": "user", "content": message_text}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return f"Error: {e}"

# 6. Asking for JSON: Structured Order Extraction
def extract_order_json(customer_message: str) -> dict:
    """Extracts customer order details into clean structured JSON."""
    system_prompt = """Extract the bakery order details from the customer message.
Return ONLY valid JSON with no markdown formatting or extra text.
Required keys:
- "name": string (customer name, or null if not provided)
- "item": string (bakery item name)
- "qty": integer (number of units)
- "date": string (requested delivery/pickup date)
- "eggless": boolean (true/false)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": customer_message}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1
        )
        raw_text = response.choices[0].message.content.strip()
        # Clean any accidental code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": raw_text}
    except Exception as e:
        return {"error": str(e)}

# 7. Multi-Turn Memory: Sweet Crumbs Conversational Chatbot
SYSTEM_PROMPT = f"""ROLE: You are the friendly, helpful assistant for Sweet Crumbs Home Bakery.

FACTS:
- Current Menu & Prices (in INR): {json.dumps(MENU)}
- We bake fresh to order. Orders require minimum 24 hours' notice.
- ALL cakes can be baked in 100% eggless versions upon request.

RULES:
- NEVER guarantee or promise an exact delivery time.
- If an item or detail is not in FACTS, say: "I'll check with the owner."
- Remember the customer's name, ordered items, and preferences throughout the conversation.

FORMAT:
- Keep replies friendly, concise (under 50 words), and match the customer's language (English or Telugu)."""

class BakeryChatbot:
    """Chatbot maintaining conversational memory with sliding-window history trimming."""
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> str:
        # Append user message
        self.history.append({"role": "user", "content": user_text})

        # Sliding window trimming: keep system prompt + last N messages to control token limits
        recent_context = [self.history[0]] + self.history[1:][-self.max_turns:]

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=recent_context,
                temperature=0.2,
                max_tokens=150
            )
            reply = response.choices[0].message.content.strip()
        except RateLimitError:
            reply = "We are currently experiencing high traffic. Please try again in a moment."
        except AuthenticationError:
            reply = "Setup issue: Authentication failed. Please check the API configuration."
        except APIConnectionError:
            reply = "Connection error: Unable to reach the bakery server. Please check your network."
        except Exception as e:
            reply = f"An unexpected error occurred: {str(e)}"

        # Append assistant reply to persistent history
        self.history.append({"role": "assistant", "content": reply})
        return reply


# --- Main Demonstration ---
if __name__ == "__main__":
    print("=" * 65)
    print("DAY 2: PROMPTS, MEMORY & PYTHON TOOLS (SWEET CRUMBS BAKERY)")
    print("=" * 65)

    # Demo 1: Function, Type Hints, Decorator
    print("\n1. Function with Type Hints & @log_call Decorator:")
    total = get_price("chocolate cake", 2)
    print(f"   Calculated total: Rs {total}")

    # Demo 2: Few-Shot Intent Classification
    print("\n2. Few-Shot Intent Classification:")
    sample_msgs = [
        "Can I get 3 cupcakes tomorrow?",
        "Do you deliver to Madhapur?",
        "The bread was burnt and completely dry!"
    ]
    for msg in sample_msgs:
        category = classify_intent(msg)
        print(f"   '{msg}' -> Category: [{category}]")

    # Demo 3: Structured JSON Extraction
    print("\n3. Structured JSON Order Extraction:")
    order_text = "Hi this is Ravi, I need 2 chocolate cakes for 28th Nov, please make them eggless."
    parsed = extract_order_json(order_text)
    print(f"   Input: '{order_text}'")
    print(f"   Parsed JSON: {json.dumps(parsed, indent=2)}")

    # Demo 4: Multi-Turn Conversational Memory & Context Trimming
    print("\n4. Multi-Turn Conversational Memory:")
    bot = BakeryChatbot(max_turns=6)

    turns = [
        "Hi, my name is Ravi.",
        "Can I get 2 chocolate cakes for tomorrow? Make them eggless.",
        "What did I order so far, and what will be my total price?"
    ]

    for user_turn in turns:
        print(f"\nUser: {user_turn}")
        bot_reply = bot.chat(user_turn)
        print(f"Bot:  {bot_reply}")

    print("\n" + "=" * 65)
    print("Day 2 demonstrations completed successfully!")
    print("=" * 65)
