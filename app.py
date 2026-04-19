import os
import re
import tempfile
import subprocess
import streamlit as st
import validators
import whisper

from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# -------------------------
# ENV
# -------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Video Summarizer", page_icon="🦜")
st.title("🦜 YouTube / Website Summarizer")

url = st.text_input("Enter YouTube or Website URL")

# -------------------------
# LLM (STREAMLIT SECRETS FIX)
# -------------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.2
    )

# -------------------------
# WHISPER
# -------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# -------------------------
# HELPERS
# -------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def summarize_text(llm, text):
    prompt = f"""
You are a strict summarizer.

RULES:
- Use ONLY provided text
- Do NOT hallucinate
- Output must be bullet points

TASK:
Summarize in 5–8 bullet points.

TEXT:
{text}
"""
    return llm.invoke(prompt).content


# -------------------------
# YOUTUBE → AUDIO → WHISPER (CLEAN VERSION)
# -------------------------
def audio_to_text(url):
    try:
        tmp_dir = tempfile.mkdtemp()
        output_template = os.path.join(tmp_dir, "audio.%(ext)s")

        st.info("🎬 Processing video...")

        # ✅ simplified yt-dlp (no ffmpeg path, no fragile flags)
        cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", output_template,
            url
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        # handle YouTube block / 403
        if result.returncode != 0:
            st.error("❌ Video download failed (YouTube blocked or unavailable)")
            st.code(result.stderr[-1000:])
            return "ERROR"

        files = os.listdir(tmp_dir)
        audio_files = [f for f in files if f.endswith(".mp3")]

        if not audio_files:
            return "ERROR"

        audio_path = os.path.join(tmp_dir, audio_files[0])

        st.info("🧠 Transcribing...")

        model = load_whisper()
        result = model.transcribe(audio_path)

        return result.get("text", "").strip()

    except Exception as e:
        st.error(str(e))
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
            text = audio_to_text(url)

            if text == "ERROR":
                st.error("❌ Could not process video")
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
