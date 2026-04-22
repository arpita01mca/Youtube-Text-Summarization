import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# -------------------------
# ENV
# -------------------------
load_dotenv()

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
# YOUTUBE TRANSCRIPT (ROBUST)
# -------------------------
def get_transcript(url):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return "ERROR"

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            transcript = transcript_list.find_generated_transcript(['en'])

        data = transcript.fetch()
        text = " ".join([t["text"] for t in data])

        return text

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return "ERROR"
    except Exception:
        return "ERROR"


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
        # YOUTUBE
        # -------------------------
        if "youtube.com" in url or "youtu.be" in url:

            text = get_transcript(url)

            if text == "ERROR":
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

    except Exception as e:
        st.exception(e)
