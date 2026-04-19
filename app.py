import os
import re
import streamlit as st
import validators
import whisper

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")

# -------------------------
# LLM
# -------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.2
    )

# -------------------------
# WHISPER (FALLBACK)
# -------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# -------------------------
# CLEAN
# -------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# -------------------------
# SUMMARY
# -------------------------
def summarize_text(llm, text):
    prompt = f"""
You are a strict summarizer.

RULES:
- Use ONLY provided text
- No hallucination
- Bullet points only

TEXT:
{text}
"""
    return llm.invoke(prompt).content

# -------------------------
# TRANSCRIPT
# -------------------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except:
        return None

# -------------------------
# WHISPER FALLBACK (SIMULATED TEXT INPUT ONLY)
# -------------------------
def whisper_fallback(video_id):
    st.warning("⚠️ No transcript found. Whisper fallback required.")

    st.info("👉 This requires audio extraction (not supported on all cloud deployments).")

    return None  # keep safe for cloud

# -------------------------
# WEBSITE
# -------------------------
def load_website(url):
    loader = UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    docs = loader.load()
    return " ".join([d.page_content for d in docs])

# -------------------------
# MAIN
# -------------------------
if st.button("Summarize"):

    if not url:
        st.error("Enter URL")
        st.stop()

    llm = get_llm()
    text = None

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
        if not match:
            st.error("Invalid YouTube URL")
            st.stop()

        video_id = match.group(1)

        st.info("📄 Trying transcript...")

        text = get_transcript(video_id)

        if not text:
            st.error("❌ No transcript available for this video")
            st.warning("👉 Whisper fallback is required (not enabled in cloud-safe mode)")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("🌐 Loading website...")
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text:
        st.error("No usable content")
        st.stop()

    text = clean_text(text)[:12000]

    # -------------------------
    # PREVIEW
    # -------------------------
    st.write("### 🔍 Preview")
    st.text_area("", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    st.info("🧠 Generating summary...")
    summary = summarize_text(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
