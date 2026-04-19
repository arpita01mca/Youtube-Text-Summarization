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

url = st.text_input("Enter YouTube or Website URL")


# -------------------------
# SAFE ENV ACCESS (LOCAL + CLOUD)
# -------------------------
def get_secret(key):
    return os.getenv(key) or st.secrets.get(key, None)


# -------------------------
# LLM (FIXED)
# -------------------------
@st.cache_resource
def get_llm():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        st.error("❌ GROQ_API_KEY missing in environment or Streamlit secrets")
        st.stop()

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
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


# -------------------------
# YOUTUBE TRANSCRIPT (PRIMARY METHOD)
# -------------------------
def get_transcript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in data])

    except (TranscriptsDisabled, NoTranscriptFound):
        return None

    except Exception:
        return None


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
# SUMMARY (BULLET POINTS)
# -------------------------
def summarize(llm, text):
    prompt = f"""
You are a professional summarizer.

RULES:
- Use ONLY the given text
- No hallucination
- No external knowledge

TASK:
Convert into 5–8 clear bullet points.

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
    text = ""

    # -------------------------
    # YOUTUBE
    # -------------------------
    if "youtube.com" in url or "youtu.be" in url:

        video_id = get_video_id(url)

        st.info("📄 Fetching transcript...")

        text = get_transcript(video_id)

        if not text:
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
    if not text or len(text.strip()) < 50:
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
    summary = summarize(llm, text)

    st.success("✅ Summary Generated")
    st.write(summary)
