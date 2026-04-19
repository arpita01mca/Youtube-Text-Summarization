import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# -------------------------
# ENV
# -------------------------
load_dotenv()

st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")


# -------------------------
# INPUT
# -------------------------
url = st.text_input("Enter YouTube or Website URL")


# -------------------------
# SAFE SECRET HANDLER (Streamlit Cloud safe)
# -------------------------
def get_secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)


# -------------------------
# LLM (GROQ)
# -------------------------
@st.cache_resource
def get_llm():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY missing in environment/secrets")
        st.stop()

    return ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=api_key,
        temperature=0.2
    )


# -------------------------
# YOUTUBE ID EXTRACTION (robust)
# -------------------------
def get_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"shorts/([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


# -------------------------
# GET TRANSCRIPT
# -------------------------
@st.cache_data
def get_transcript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in data])

    except (TranscriptsDisabled, NoTranscriptFound):
        st.warning("⚠️ No captions available for this video.")
        return None

    except Exception as e:
        st.warning(f"Transcript error: {e}")
        return None


# -------------------------
# WEBSITE LOADER
# -------------------------
def load_website(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        return " ".join([d.page_content for d in docs])

    except Exception as e:
        st.warning(f"Website loading failed: {e}")
        return None


# -------------------------
# CHUNKING
# -------------------------
def chunk_text(text, size=4000):
    return [text[i:i+size] for i in range(0, len(text), size)]


# -------------------------
# SUMMARIZER (ROBUST MULTI-CHUNK)
# -------------------------
def summarize(llm, text):
    chunks = chunk_text(text)

    partial_summaries = []

    for i, chunk in enumerate(chunks):
        prompt = f"""
Summarize the following content clearly in bullet points.

Rules:
- Only use given text
- No external knowledge
- Be concise

TEXT:
{chunk}
"""
        partial = llm.invoke(prompt).content
        partial_summaries.append(partial)

    final_prompt = f"""
Combine these partial summaries into one clean final summary.

Make it:
- 5 to 10 bullet points
- well structured
- non repetitive

CONTENT:
{" ".join(partial_summaries)}
"""

    return llm.invoke(final_prompt).content


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
    text = None

    # -------------------------
    # YOUTUBE FLOW
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        video_id = get_video_id(url)

        if not video_id:
            st.error("Invalid YouTube URL")
            st.stop()

        st.info("📄 Fetching transcript...")
        text = get_transcript(video_id)

    # -------------------------
    # WEBSITE FLOW
    # -------------------------
    else:
        st.info("🌐 Loading website content...")
        text = load_website(url)

    # -------------------------
    # VALIDATION
    # -------------------------
    if not text or len(text.strip()) < 50:
        st.error("❌ No usable content found")
        st.stop()

    # -------------------------
    # PREVIEW
    # -------------------------
    st.subheader("🔍 Preview")
    st.text_area("", text[:1000], height=200)

    # -------------------------
    # SUMMARY
    # -------------------------
    with st.spinner("🧠 Generating summary..."):
        summary = summarize(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
