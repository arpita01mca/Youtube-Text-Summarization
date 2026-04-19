import os
import re
import streamlit as st
import validators

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
# LLM (STREAMLIT SECRETS)
# -------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
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
- Do NOT hallucinate
- Output must be bullet points

TASK:
Summarize in 5–8 bullet points.

TEXT:
{text}
"""
    return llm.invoke(prompt).content

# -------------------------
# YOUTUBE TRANSCRIPT (FIXED)
# -------------------------
def get_youtube_text(url):
    try:
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
        if not match:
            return "ERROR"

        video_id = match.group(1)

        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        text = " ".join([t["text"] for t in transcript])
        return text

    except Exception as e:
        return f"ERROR: {str(e)}"

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
# MAIN APP
# -------------------------
if st.button("Summarize"):

    if not url:
        st.error("Enter URL")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    llm = get_llm()

    try:
        # -------------------------
        # YOUTUBE (FIXED APPROACH)
        # -------------------------
        if "youtube.com" in url or "youtu.be" in url:

            st.info("📄 Fetching transcript...")

            text = get_youtube_text(url)

            if text.startswith("ERROR"):
                st.error("❌ No transcript available for this video")
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
        if not text or len(text) < 50:
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

    except Exception as e:
        st.exception(e)
