import os
import re
import streamlit as st
import validators
import requests
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

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
# CLEAN
# -------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# -------------------------
# YOUTUBE ID
# -------------------------
def get_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

# -------------------------
# TRANSCRIPT API
# -------------------------
def get_transcript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in data])
    except:
        return None

# -------------------------
# 🔥 REAL WHISPER API
# -------------------------
def whisper_api(url):
    api_key = os.getenv("WHISPER_API_KEY")

    if not api_key:
        st.error("❌ WHISPER_API_KEY missing in .env")
        return None

    try:
        st.info("🧠 Sending video to Whisper API...")

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            data={
                "model": "whisper-1",
                "url": url
            }
        )

        if response.status_code == 200:
            return response.json().get("text")

        st.error(f"Whisper API error: {response.text}")
        return None

    except Exception as e:
        st.error(str(e))
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

        st.info("📄 Trying transcript...")

        text = get_transcript(video_id)

        # -------------------------
        # WHISPER FALLBACK (REAL)
        # -------------------------
        if not text:
            st.warning("No captions found → using Whisper API...")

            text = whisper_api(url)

            if not text:
                st.error("❌ Whisper API failed or not configured")
                st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text:
        st.error("No content found")
        st.stop()

    text = clean_text(text)[:12000]

    st.write("### Preview")
    st.text_area("", text[:1000], height=200)

    summary = summarize(llm, text)

    st.success("Summary Generated")
    st.write(summary)
