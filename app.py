import os
import streamlit as st
import validators
import yt_dlp
import whisper
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# -------------------------
# ENV
# -------------------------
load_dotenv()

st.set_page_config(page_title="Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")


# -------------------------
# LLM
# -------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )


# -------------------------
# LOAD WHISPER MODEL (cached)
# -------------------------
@st.cache_resource
def load_model():
    return whisper.load_model("base")


# -------------------------
# YOUTUBE TRANSCRIPTION (WHISPER)
# -------------------------
def transcribe_youtube(url):
    try:
        audio_file = "audio.mp3"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio.%(ext)s",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        model = load_model()
        result = model.transcribe(audio_file)

        if os.path.exists(audio_file):
            os.remove(audio_file)

        return result["text"]

    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


# -------------------------
# WEBSITE SCRAPER
# -------------------------
def get_website_text(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.error(f"Website error: {e}")
        return None


# -------------------------
# SUMMARIZER
# -------------------------
def summarize(llm, text):
    prompt = f"""
You are a strict summarizer.

RULES:
- Only use provided text
- Do NOT hallucinate

TASK:
Give 5–8 bullet points summary.

TEXT:
{text}
"""
    return llm.invoke(prompt).content


# -------------------------
# MAIN
# -------------------------
if st.button("Summarize"):

    if not url:
        st.error("Please enter a URL")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    llm = get_llm()
    text = None

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        st.info("🎧 Downloading & transcribing video (30–60s)...")

        text = transcribe_youtube(url)

        if not text:
            st.error("❌ Could not transcribe video")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("🌐 Fetching website content...")
        text = get_website_text(url)

        if not text:
            st.error("❌ Could not load website")
            st.stop()

    # -------------------------
    # VALIDATION
    # -------------------------
    if len(text.strip()) < 20:
        st.error("No usable content found")
        st.stop()

    text = text[:12000]

    st.write("### 🔍 Preview")
    st.text_area("Content", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    with st.spinner("Summarizing..."):
        summary = summarize(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
