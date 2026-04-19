import os
import re
import streamlit as st
import validators
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# -------------------------
# ENV
# -------------------------
load_dotenv()

st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")


# -------------------------
# SAFE SECRET HANDLER
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
        st.error("❌ GROQ_API_KEY missing")
        st.stop()

    return ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=api_key,
        temperature=0.2
    )


# -------------------------
# YOUTUBE ID EXTRACTION
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
# YOUTUBE TRANSCRIPT (FIXED API)
# -------------------------
@st.cache_data
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript()
        except:
            transcript = transcript_list.find_generated_transcript(['en'])

        data = transcript.fetch()
        return " ".join([x["text"] for x in data])

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
# CHUNK TEXT
# -------------------------
def chunk_text(text, size=4000):
    return [text[i:i+size] for i in range(0, len(text), size)]


# -------------------------
# SUMMARIZER (ROBUST)
# -------------------------
def summarize(llm, text):
    chunks = chunk_text(text)

    partial_summaries = []

    for chunk in chunks:
        prompt = f"""
Summarize the following text in clear bullet points.

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
Combine these into one clean summary.

Rules:
- 5–10 bullet points
- Remove repetition
- Keep clarity

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
        st.info("🌐 Loading website...")
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
