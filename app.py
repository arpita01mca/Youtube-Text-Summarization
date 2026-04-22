import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi

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
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

# -------------------------
# HELPERS
# -------------------------
def get_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def summarize_text(llm, text):
    prompt = f"""
You are a strict summarizer.

RULES:
- Use ONLY provided text
- Do NOT hallucinate

TASK:
Write a bullet-point summary (5–8 points).

TEXT:
{text}
"""
    return llm.invoke(prompt).content


# -------------------------
# TRANSCRIPT ONLY (CLOUD SAFE)
# -------------------------
def get_transcript(url):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return None

        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in data])

    except:
        return None


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

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    llm = get_llm()

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        st.info("Fetching transcript...")

        text = get_transcript(url)

        if not text:
            st.error("❌ No transcript available for this video")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text or len(text) < 50:
        st.error("No usable content found")
        st.stop()

    text = text[:12000]

    st.write("### 🔍 Preview")
    st.text_area("", text[:1000], height=200)

    summary = summarize_text(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
