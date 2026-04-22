import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

import yt_dlp

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
        temperature=0.2
    )


# -------------------------
# VIDEO ID
# -------------------------
def get_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ]

    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)

    return None


# -------------------------
# METHOD 1: transcript API
# -------------------------
def get_transcript_api(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = next(iter(transcript_list))

        data = transcript.fetch()
        return " ".join([t["text"] for t in data])

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        return None


# -------------------------
# METHOD 2: yt-dlp fallback (IMPORTANT FIX)
# -------------------------
def get_transcript_fallback(url):
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles") or info.get("automatic_captions")

        if not subs or "en" not in subs:
            return None

        entries = subs["en"]

        text = []
        for item in entries:
            if isinstance(item, dict):
                text.append(item.get("text", ""))
            else:
                text.append(str(item))

        return " ".join(text)

    except Exception as e:
        print("yt-dlp error:", e)
        return None


# -------------------------
# UNIFIED TRANSCRIPT
# -------------------------
def get_transcript(url):
    video_id = get_video_id(url)
    if not video_id:
        return None

    text = get_transcript_api(video_id)

    if not text:
        text = get_transcript_fallback(url)

    return text


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
# SUMMARIZER
# -------------------------
def summarize_text(llm, text):
    prompt = f"""
You are a strict summarizer.

RULES:
- Use ONLY provided text
- Do NOT add external knowledge
- Do NOT hallucinate

TASK:
Write 5–8 bullet points summarizing the content.

TEXT:
{text}
"""
    return llm.invoke(prompt).content


# -------------------------
# MAIN APP
# -------------------------
if st.button("Summarize"):

    if not url:
        st.error("Please enter a URL")
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

        if not text or len(text.strip()) < 20:
            st.error("❌ No transcript available for this video")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("Loading website...")
        text = load_website(url)

    # -------------------------
    # FINAL CHECK
    # -------------------------
    if not text or len(text.strip()) < 20:
        st.error("No usable content found")
        st.stop()

    text = text[:12000]

    st.write("### 🔍 Preview")
    st.text_area("Content preview", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    with st.spinner("Summarizing..."):
        summary = summarize_text(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
