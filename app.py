import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# -------------------------
# ENV
# -------------------------
load_dotenv()

st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")

# -------------------------
# LLM
# -------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.stop()

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.2
    )

# -------------------------
# CLEAN TEXT
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
- Return bullet points (5–8)

TEXT:
{text}
"""
    return llm.invoke(prompt).content

# -------------------------
# GET VIDEO ID
# -------------------------
def get_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# -------------------------
# TRANSCRIPT FETCH
# -------------------------
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return None

# -------------------------
# WEBSITE LOADER
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
        st.error("Please enter URL")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    llm = get_llm()
    text = ""

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        video_id = get_video_id(url)

        if not video_id:
            st.error("Invalid YouTube URL")
            st.stop()

        st.info("📄 Fetching transcript...")

        text = get_transcript(video_id)

        if not text:
            st.error("❌ No transcript available for this video")

            st.warning(
                "This video has no captions. "
                "To support all videos, you need Whisper API or backend audio extraction."
            )
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("🌐 Loading website content...")
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text or len(text.strip()) < 50:
        st.error("No usable content found")
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
