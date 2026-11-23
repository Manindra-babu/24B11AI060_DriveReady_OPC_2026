import streamlit as st
from groq import Groq
import os
from pathlib import Path
from bakery_db import BAKERY_MENU, BAKERY_INFO, SYSTEM_PROMPT, SAMPLE_QUESTIONS

# Page configuration
st.set_page_config(
    page_title="Sweet Crumbs Bakery Assistant",
    page_icon="🧁",
    layout="wide"
)

# Custom minimal styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #8B4513;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .status-badge {
        background-color: #e6f4ea;
        color: #137333;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    .stButton>button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# Load API Key securely from environment or .env file (NEVER hardcoded or exposed in UI)
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment or .env file. Please configure it in your .env file.")
    st.stop()

MODEL_NAME = "openai/gpt-oss-120b"

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Welcome to Sweet Crumbs Bakery & Cafe 🧁. How can I assist you with our fresh cakes, breads, or order policies today?"}
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --- SIDEBAR (Left Panel) ---
with st.sidebar:
    st.title("🥖 Sweet Crumbs Bakery")
    st.caption("Artisanal Bakery & Cafe • Freshly baked to order")
    
    st.markdown(f"""
    **⏰ Hours:** {BAKERY_INFO['hours']}  
    **📍 Location:** {BAKERY_INFO['location']}  
    **📞 Contact:** {BAKERY_INFO['phone']}
    """)
    st.divider()

    # Basic Suggested Questions Section
    st.subheader("💡 Suggested Questions")
    st.caption("Click any question to ask instantly:")
    for idx, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(f"❓ {q}", key=f"sample_q_{idx}"):
            st.session_state.pending_question = q

    st.divider()

    # Simple Database & Categories Browser
    st.subheader("📋 Menu Database")
    category = st.selectbox("Select Category:", list(BAKERY_MENU.keys()))
    
    items = BAKERY_MENU[category]
    for item in items:
        with st.expander(f"{item['name']} — {item['price']}"):
            st.write(f"**Dietary:** {item['dietary']}")
            st.write(f"**Description:** {item['desc']}")

    st.divider()

    # Engine Status (Secure - No API Key shown)
    st.markdown('<div class="status-badge">🟢 Agent Online: openai/gpt-oss-120b</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Welcome to Sweet Crumbs Bakery & Cafe 🧁. How can I assist you today?"}
        ]
        st.rerun()

# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">🧁 Sweet Crumbs Bakery Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask about our daily bakes, eggless cakes, prices, ingredients, and delivery policies.</div>', unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Streaming response generator for ultra-fast real-time output
def stream_bakery_agent(prompt_text):
    client = Groq(api_key=GROQ_API_KEY)
    
    # Build minimal prompt messages
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Keep only last 4 context messages to avoid latency buildup
    for m in st.session_state.messages[-4:]:
        api_messages.append({"role": m["role"], "content": m["content"]})
        
    api_messages.append({"role": "user", "content": prompt_text})
    
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_messages,
        temperature=0.2,
        max_tokens=400,
        stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Determine question input
user_input = st.chat_input("Ask a question about our bakery...")

# If clicked from sidebar questions
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

# Process User Query
if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Stream assistant response instantly
    with st.chat_message("assistant"):
        response = st.write_stream(stream_bakery_agent(user_input))
            
    st.session_state.messages.append({"role": "assistant", "content": response})
