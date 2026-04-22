import os
import re
import streamlit as st
import validators
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi

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
# YOUTUBE ID
# -------------------------
def get_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]

    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)

    return None


# -------------------------
# YOUTUBE TRANSCRIPT (SAFE)
# -------------------------
def get_youtube_text(url):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return None

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            transcript = next(iter(transcript_list))

        data = transcript.fetch()

        return " ".join([t["text"] for t in data])

    except Exception as e:
        print("Transcript error:", e)
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
        print("Website error:", e)
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
        st.error("Enter a URL")
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

        st.info("Fetching YouTube transcript...")

        text = get_youtube_text(url)

        if not text:
            st.error("❌ No transcript available for this video")
            st.stop()

    # -------------------------
    # WEBSITE
    # -------------------------
    else:
        st.info("Fetching website content...")

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

    st.write("### Preview")
    st.text_area("Content", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    with st.spinner("Summarizing..."):
        summary = summarize(llm, text)

    st.success("Done")
    st.write(summary)
