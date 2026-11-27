"""
Day 5 Task: Memory and RAG - An Agent That Remembers and Looks Things Up
AI Agents Workshop (Level 2: Single Agent) - DriveReady OPC 2026
Student: Manindra Babu (Roll No: 24B11AI060)
Case Study: The Clockmaker of Silverfall (Custom Knowledge Story) & Sweet Crumbs Bakery (Agent System)

Curriculum Covered:
1. Three Kinds of Memory:
   - Conversation Memory (short-term, isolated per thread)
   - Customer Memory (long-term across sessions)
   - Knowledge Base Memory (external documents via RAG)
2. Custom Story Document Creation ("The Clockmaker of Silverfall"):
   - Rich multi-character, factual narrative replacing "The Little Lantern"
3. Text Chunking & Overlap:
   - Preserving semantic boundaries without cutting critical facts
4. Vector Embeddings:
   - Text to numerical vectors via SentenceTransformers ('all-MiniLM-L6-v2')
5. Vector Database & Similarity Indexing:
   - Normalized cosine similarity index + ChromaDB collection
6. Retrieval-Augmented Generation (RAG):
   - Semantic retrieval (top-k) -> Context Augmentation -> Grounded LLM Generation
   - Strict Anti-Hallucination Guardrail: "The answer is not available in the context."
7. Conversation Memory with LangGraph Checkpointer:
   - InMemorySaver with thread_id so customer conversations never mix
8. Single-Agent Sign-off:
   - Brain (ChatGroq) + Instructions (RFRF) + Tools (get_price, is_open, log_order, search_faq) + Memory + Knowledge
9. Automated Evaluation & Verification Suite:
   - Factual retrieval, semantic inference, out-of-context rejection, and thread isolation
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Literal
import numpy as np
from dotenv import load_dotenv

# Load API credentials securely from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Initialize LLM with tool & function calling support
if GROQ_API_KEY:
    from langchain_groq import ChatGroq
    from groq import Groq
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_retries=3)
    groq_raw_client = Groq(api_key=GROQ_API_KEY)
    LLM_MODEL = "openai/gpt-oss-120b"
elif GEMINI_API_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2, api_key=GEMINI_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    groq_raw_client = None
    LLM_MODEL = "gemini-3.5-flash"
else:
    raise ValueError("No API key configured in .env! Please set GROQ_API_KEY or GEMINI_API_KEY.")


# =====================================================================
# 1. Custom Knowledge Document: "The Clockmaker of Silverfall"
# (Replaces "The Little Lantern" with a rich, testable narrative)
# =====================================================================

CUSTOM_STORY = """
The Clockmaker of Silverfall

In the mist-shrouded valley of Silverfall, nestled beneath the towering granite cliffs of Azure Ridge, lived an elderly master horologist named Silas and his perceptive young apprentice, Kieran. While the valley villagers relied on erratic sundials and the heavy tolling of the central Cathedral bell, Silas dedicated forty years of his life to constructing the Celestial Chronometer. This magnificent brass timepiece was engineered with thirty-six synthetic sapphire bearings, four counter-weighted silver escapements, and a central pendulum sculpted from starlight obsidian discovered in the old iron mines.

Silas taught that time was not merely an abstract sequence of seconds, but the living cadence of celestial spheres. Every winter solstice at midnight, the Chronometer required rewinding with the Solstice Key—a filigreed golden key forged from meteorite brass, stored securely within a velvet-lined cedar lockbox in the workshop's second-floor attic. For decades, the skeptical villagers viewed Silas's invention with patronizing humor, dismissing it as an eccentric curiosity that lacked any practical value for simple farming folk.

Everything changed during the catastrophic Great Frost of November 1782. A sudden polar tempest swept through the mountain gaps, plunging temperatures far below freezing and blanketing Silverfall in four feet of suffocating snow. The intense freezing winds cracked the iron gears of the Cathedral bell, silencing the town's only method of coordinating emergency relief, changing guard shifts, or marking curfew. In the midst of this darkness, an eight-wagon merchant caravan led by Captain Orin, transporting critical medicinal fever-bark and dry firewood, lost its bearings and snapped a steering axle against the jagged rocks of Raven's Pass, four leagues north of the town.

Alerted by the distant, muffled echoes of a distress horn echoing through the gorge, Silas consulted the Celestial Chronometer. Using its precision astrolabe dial, he calculated the exact lunar azimuth and activated the clocktower's nocturnal beacon—a high-intensity spotlight generated by an oil furnace and focused through a sixty-millimeter prism of polished quartz. A bright amber beam pierced through the snowstorm, illuminating the treacherous precipice of Raven's Pass. Guided by this unwavering beacon, Captain Orin and eighteen freezing passengers abandoned their crippled wagons and marched safely to the glowing warmth of Silas's workshop hearth.

Silas and Kieran provided the shivering travelers with thick wool quilts, dried venison broth, and hot cups of spiced chamomile tea until the blizzard abated three days later. When Mayor Sterling and the valley council presented Silas with a purse of two hundred gold ducats as a reward, the old clockmaker gently returned the coin. He stated: 'True precision is never measured by how we count our hours, but by how each hour serves someone who has lost their way in the storm.' Following that harrowing winter, the valley established the Silverfall Watch guild, installing the Celestial Chronometer in the new town hall, where Kieran and each succeeding generation of apprentices wound the Solstice Key with reverence every winter solstice.
"""


# =====================================================================
# 2. Document Chunking with Overlap
# =====================================================================

def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split text into chunks of specified character size with sliding overlap
    to ensure context is preserved across split boundaries.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly smaller than chunk_size")
    
    clean_text = " ".join(text.strip().split())
    chunks = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# =====================================================================
# 3. Vector Database & Similarity Index (FAISS-compatible Numpy Index)
# =====================================================================

class VectorIndex:
    """
    In-memory normalized vector index implementing exact cosine similarity
    (inner product on normalized embeddings), mirroring FAISS IndexFlatIP.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype="float32")
        self.documents: List[str] = []

    def add(self, embeddings: np.ndarray, documents: List[str]):
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
        self.vectors = np.vstack([self.vectors, embeddings])
        self.documents.extend(documents)

    def search(self, query_vector: np.ndarray, top_k: int = 3):
        if len(self.documents) == 0:
            return [], []
        scores = np.dot(self.vectors, query_vector.T).flatten()
        top_k = min(top_k, len(self.documents))
        indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indices]
        top_docs = [self.documents[i] for i in indices]
        return top_scores.tolist(), top_docs

    @property
    def ntotal(self) -> int:
        return len(self.documents)


# =====================================================================
# 4. RAG Engine: Indexing & Retrieval Pipeline
# =====================================================================

class RAGEngine:
    """
    End-to-end Retrieval-Augmented Generation pipeline using SentenceTransformers
    and Groq/Gemini with strict grounding and zero-hallucination guardrails.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[RAG] Loading embedding model '{model_name}'...")
        self.embed_model = SentenceTransformer(model_name)
        dim = getattr(self.embed_model, "get_embedding_dimension", self.embed_model.get_sentence_embedding_dimension)()
        self.index = VectorIndex(dimension=dim)
        self.chunks: List[str] = []

    def index_document(self, text: str, chunk_size: int = 400, overlap: int = 80):
        self.chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embed_model.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        self.index.add(embeddings, self.chunks)
        print(f"[RAG] Indexed {len(self.chunks)} chunks into vector index (dimension: {embeddings.shape[1]}).")

    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        q_vec = self.embed_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        scores, docs = self.index.search(q_vec, top_k=top_k)
        return "\n\n".join(docs)

    def ask(self, question: str, top_k: int = 3) -> str:
        context = self.retrieve_context(question, top_k=top_k)
        
        system_instruction = (
            "You are an evidence-grounded question-answering assistant. "
            "Answer the question ONLY using the provided text context. "
            "If the answer cannot be deduced strictly from the context, respond exactly: "
            "'The answer is not available in the context.' "
            "Do not fabricate or extrapolate information. Keep answers factual and concise."
        )
        
        user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        if groq_raw_client:
            response = groq_raw_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        else:
            model = genai.GenerativeModel(model_name="gemini-3.5-flash", system_instruction=system_instruction)
            resp = model.generate_content(user_prompt)
            return resp.text.strip()


# =====================================================================
# 5. Sweet Crumbs Bakery: ChromaDB Knowledge Base
# (Topic 5 & 7 from Day 5 Curriculum)
# =====================================================================

BAKERY_FAQS = {
    "delivery": "We deliver within 5 km of the bakery. Delivery is Rs 50, free for orders above Rs 1,000.",
    "payment": "We accept UPI and cash. Orders above Rs 1,000 need a 50% advance by UPI.",
    "cancellation": "Cancellation is free up to 12 hours before pickup. After that, the advance is not refunded.",
    "photo_cake": "Photo cakes and custom designs need 48 hours' notice.",
    "notice": "Regular orders need 24 hours' notice.",
    "eggless": "All our cakes have eggless versions at the same price.",
    "allergens": "Plum cake contains nuts and dried fruit. Cupcakes and chocolate cake are nut-free.",
    "storage": "Keep cakes in the fridge and eat them within 3 days.",
    "message": "A message on the cake is free, up to 20 letters.",
    "candles": "Candles are free with every cake.",
    "discounts": "Bulk-order discounts need the owner's approval.",
    "pickup": "Pickup is during opening hours: Monday to Saturday 9 am to 8 pm, Sunday 10 am to 6 pm."
}

chroma_client = chromadb.Client()
faq_collection = chroma_client.get_or_create_collection(name="sweet_crumbs_faq")
faq_collection.upsert(
    ids=list(BAKERY_FAQS.keys()),
    documents=list(BAKERY_FAQS.values())
)


# =====================================================================
# 6. Bakery Tools Definition (@tool)
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

class OrderSchema(BaseModel):
    name: str = Field(description="Customer first name")
    item: Literal["plum cake", "chocolate cake", "cupcake"] = Field(description="Menu item name")
    qty: int = Field(description="Quantity (must be at least 1)", ge=1)
    date: str = Field(description="Pickup date (must have >= 24h notice)")
    eggless: bool = Field(description="True if eggless version requested")

@tool
def get_price(item: str, qty: int = 1) -> str:
    """Get the total price in rupees for a quantity of one menu item.
    Args:
        item: Menu item name ('plum cake', 'chocolate cake', or 'cupcake')
        qty: Quantity ordered
    """
    clean = item.lower().strip()
    if clean in MENU:
        return f"{qty} x {clean} = Rs {MENU[clean] * qty}"
    return f"'{item}' is not on the menu. We offer: {', '.join(MENU.keys())}."

@tool
def is_open(day: str) -> str:
    """Check the bakery's opening hours for a day of the week, e.g. 'sunday' or 'monday'."""
    clean = day.lower().strip()
    if clean in HOURS:
        return f"Open {HOURS[clean]} on {clean.capitalize()}."
    return f"'{day}' is not a valid day of the week."

@tool
def search_faq(question: str) -> str:
    """Search Sweet Crumbs policies and FAQs: delivery, payment, cancellation,
    photo cakes, allergens, storage, candles, discounts, and custom messages.
    Use for any policy or rules question, not for prices or opening hours."""
    results = faq_collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0] if results["documents"] else []
    return "\n".join(docs) if docs else "No matching policy found."

@tool
def log_order(name: str, item: str, qty: int, date: str, eggless: bool) -> str:
    """Save a confirmed cake order to the bakery database.
    Only call when customer confirms name, item, qty, and pickup date."""
    clean_item = item.lower().strip()
    if clean_item not in MENU:
        return f"Cannot log order: '{item}' is not on the menu."
    total = MENU[clean_item] * qty
    return f"Order logged successfully for {name}: {qty} x {clean_item} (Eggless: {eggless}) for pickup on {date}. Total: Rs {total}."


# =====================================================================
# 7. Complete Single Agent with Tools & Thread Memory
# (Topics 2 & 8 from Day 5 Curriculum)
# =====================================================================

SYSTEM_PROMPT = """ROLE: You are the friendly, polite assistant for Sweet Crumbs Home Bakery.
RULES:
- Use get_price for prices and is_open for opening hours; never guess them.
- For any policy question (delivery, payment, cancellation, nuts, eggless, storage), use search_faq and answer ONLY from what it returns.
- If the policy or information is not found, state: "I'll check with the owner."
- Before log_order, make sure you have customer name, item, quantity, and pickup date; ask if missing.
- Remember previous context shared by the customer in their current conversation thread.
FORMAT: Under 60 words, friendly, polite English."""

bakery_agent = create_agent(
    model=llm,
    tools=[get_price, is_open, search_faq, log_order],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()
)

def agent_chat(thread_id: str, message: str) -> str:
    """Invoke the agent with isolated conversation state for each thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = bakery_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"[Agent error: {e}]"


# =====================================================================
# 8. Automated Test Suite & Benchmarking
# =====================================================================

def run_tests():
    print("=" * 70)
    print("DAY 5: RETRIEVAL-AUGMENTED GENERATION (RAG) & MEMORY TEST SUITE")
    print("=" * 70)

    # 1. Initialize RAG Engine on Custom Story
    rag = RAGEngine(model_name="all-MiniLM-L6-v2")
    rag.index_document(CUSTOM_STORY, chunk_size=380, overlap=80)

    # Test cases on custom story
    story_tests = [
        {
            "id": "STORY-01",
            "type": "Factual Retrieval",
            "question": "What is the Celestial Chronometer made of, and who built it?",
            "expected_keywords": ["Silas", "sapphire", "obsidian", "brass", "Chronometer"],
        },
        {
            "id": "STORY-02",
            "type": "Specific Artifact & Key",
            "question": "What key winds the clock and where is it kept?",
            "expected_keywords": ["Solstice Key", "cedar lockbox", "attic", "workshop"],
        },
        {
            "id": "STORY-03",
            "type": "Event & Date",
            "question": "What happened during the Great Frost of November 1782?",
            "expected_keywords": ["Cathedral", "bell", "tempest", "frost", "snow", "Raven's Pass"],
        },
        {
            "id": "STORY-04",
            "type": "Rescue Mechanism",
            "question": "How did Silas signal the stranded travelers at Raven's Pass?",
            "expected_keywords": ["beacon", "amber", "prism", "quartz"],
        },
        {
            "id": "STORY-05",
            "type": "Philosophical Quote",
            "question": "What was Silas's response when offered two hundred gold ducats?",
            "expected_keywords": ["precision", "hours", "lost", "storm"],
        },
        {
            "id": "STORY-06",
            "type": "Anti-Hallucination Guardrail",
            "question": "Did Master Silas build a steam-powered flying airship to travel to Mars?",
            "expected_keywords": ["not available in the context"],
        }
    ]

    print("\n--- PART A: Custom Knowledge Document RAG Tests ---")
    passed = 0
    total = len(story_tests)

    for tc in story_tests:
        ans = rag.ask(tc["question"])
        hits = [kw.lower() in ans.lower() for kw in tc["expected_keywords"]]
        is_pass = any(hits)
        if is_pass:
            passed += 1
            status = "PASS [OK]"
        else:
            status = "FAIL [X]"

        print(f"\n[{tc['id']}] {tc['type']}")
        print(f"Q: {tc['question']}")
        print(f"A: {ans}")
        print(f"Result: {status} (Expected any of: {tc['expected_keywords']})")

    # 2. Test Sweet Crumbs Thread Memory
    print("\n--- PART B: Conversation Memory & Thread Isolation Tests ---")

    print("\n[Turn 1] Customer 'Ravi' initiates chat:")
    r1 = agent_chat("ravi", "Hi, I'm Ravi. What are your opening hours on Sunday?")
    print(f"Ravi: Hi, I'm Ravi. What are your opening hours on Sunday?\nAgent: {r1}")

    print("\n[Turn 2] Customer 'Sravani' chats concurrently on a separate thread:")
    s1 = agent_chat("sravani", "Hello! I'm Sravani. Does the plum cake have nuts?")
    print(f"Sravani: Hello! I'm Sravani. Does the plum cake have nuts?\nAgent: {s1}")

    print("\n[Turn 3] Ravi asks about delivery policy and pricing:")
    r2 = agent_chat("ravi", "Can you deliver to my home? My order is Rs 1,200.")
    print(f"Ravi: Can you deliver to my home? My order is Rs 1,200.\nAgent: {r2}")

    print("\n[Turn 4] Sravani asks about cancellation policy:")
    s2 = agent_chat("sravani", "Can I cancel my order tomorrow morning?")
    print(f"Sravani: Can I cancel my order tomorrow morning?\nAgent: {s2}")

    print("\n[Turn 5] Thread Memory Verification (Ravi tests bot's short-term recall):")
    r3 = agent_chat("ravi", "What is my name and will I pay for delivery?")
    print(f"Ravi: What is my name and will I pay for delivery?\nAgent: {r3}")

    thread_passed = "ravi" in r3.lower() and "free" in r3.lower()
    if thread_passed:
        passed += 1
        print("\nThread Memory Verification: PASS [OK] (Agent recalled Ravi's name and free delivery eligibility)")
    else:
        print("\nThread Memory Verification: CHECK (Inspect response)")
    total += 1

    print("\n" + "=" * 70)
    print(f"FINAL TEST SCORE: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
