import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

load_dotenv()

st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")


# -------------------------
# ENV
# -------------------------
def get_secret(key):
    return os.getenv(key) or st.secrets.get(key, None)


# -------------------------
# LLM
# -------------------------
@st.cache_resource
def get_llm():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        st.error("Missing GROQ_API_KEY")
        st.stop()

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.2
    )


# -------------------------
# VIDEO ID
# -------------------------
def get_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


# -------------------------
# TRANSCRIPT (PRIMARY)
# -------------------------
def get_transcript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in data])
    except:
        return None


# -------------------------
# WHISPER API (OPTIONAL FALLBACK)
# -------------------------
def whisper_api_transcribe(video_url):
    """
    ⚠️ Placeholder for real Whisper API (OpenAI / AssemblyAI)
    """
    api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        return None

    # NOTE: You would implement actual API call here
    # returning None for now if not configured
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
# SUMMARY
# -------------------------
def summarize(llm, text):
    prompt = f"""
Summarize in 5–8 bullet points:

{text}
"""
    return llm.invoke(prompt).content


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
    text = ""

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        video_id = get_video_id(url)

        st.info("📄 Fetching transcript...")

        text = get_transcript(video_id)

        # fallback
        if not text:
            st.warning("No captions found → trying Whisper API fallback")

            text = whisper_api_transcribe(url)

        if not text:
            st.error("❌ No transcript available (enable Whisper API for full support)")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        text = load_website(url)

    # -------------------------
    # CLEAN
    # -------------------------
    text = re.sub(r"\s+", " ", text).strip()[:12000]

    st.write("### Preview")
    st.text_area("", text[:1000], height=200)

    summary = summarize(llm, text)

    st.success("Summary Generated")
    st.write(summary)
