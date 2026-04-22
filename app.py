import os
import re
import streamlit as st
import validators
import yt_dlp
import whisper
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# -------------------------
# CONFIG
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
        temperature=0.2
    )


# -------------------------
# VIDEO DOWNLOAD + TRANSCRIBE (WHISPER)
# -------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


def transcribe_video(url):
    try:
        audio_file = "audio.mp3"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio.%(ext)s",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        model = load_whisper()
        result = model.transcribe(audio_file)

        if os.path.exists(audio_file):
            os.remove(audio_file)

        return result["text"]

    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


# -------------------------
# WEBSITE LOADER (simple fallback)
# -------------------------
def load_website(url):
    import requests
    from bs4 import BeautifulSoup

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    return soup.get_text(separator=" ", strip=True)


# -------------------------
# SUMMARIZER
# -------------------------
def summarize(llm, text):
    prompt = f"""
You are a strict summarizer.

Rules:
- Only use provided text
- Do not hallucinate

Task:
Give 5–8 bullet points summary.

Text:
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

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        st.info("Downloading & transcribing video (this may take ~30–60s)...")

        text = transcribe_video(url)

        if not text:
            st.error("❌ Could not transcribe video")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("Loading website...")
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text or len(text.strip()) < 20:
        st.error("No usable content found")
        st.stop()

    text = text[:12000]

    st.write("### Preview")
    st.text_area("Content", text[:1000], height=200)

    with st.spinner("Summarizing..."):
        summary = summarize(llm, text)

    st.success("Done")
    st.write(summary)
