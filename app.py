import os
import re
import tempfile
import subprocess
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


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


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
# 1. TRY TRANSCRIPT
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
# 2. WHISPER FALLBACK (yt-dlp)
# -------------------------
def get_audio_transcript(url):
    try:
        import whisper

        tmp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(tmp_dir, "audio.mp3")

        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", audio_path,
            url
        ]

        subprocess.run(cmd, check=True)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)

        return result["text"]

    except Exception as e:
        st.error(f"Whisper failed: {e}")
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
    text = None

    # -------------------------
    # YOUTUBE PIPELINE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        st.info("Trying transcript API...")

        text = get_transcript(url)

        if not text:
            st.warning("Transcript not available → using Whisper fallback...")
            text = get_audio_transcript(url)

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

    text = clean_text(text)[:12000]

    # -------------------------
    # PREVIEW
    # -------------------------
    st.write("### 🔍 Preview")
    st.text_area("", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    summary = summarize_text(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
