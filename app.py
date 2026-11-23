import os
import re
from typing import List

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


st.set_page_config(
    page_title="YT RAG Chat",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {https://aistudio.google.com/docs/rate-limits
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.14), transparent 24%),
                radial-gradient(circle at top right, rgba(244, 114, 182, 0.12), transparent 20%),
                linear-gradient(180deg, #0b1020 0%, #0f172a 46%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.86);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        .hero-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 24px 80px rgba(2, 6, 23, 0.35);
        }
        .surface-card {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 20px;
            padding: 1rem 1.05rem;
        }
        .muted {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .tiny {
            color: #cbd5e1;
            font-size: 0.82rem;
        }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.7) !important;
            color: #e5e7eb !important;
            border-color: rgba(148, 163, 184, 0.18) !important;
        }
        .stTextInput input::placeholder {
            color: #94a3b8 !important;
        }
        .stButton button {
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: white;
            border: 0;
            border-radius: 999px;
            padding: 0.6rem 1rem;
        }
        .stButton button:hover {
            filter: brightness(1.05);
        }
        .stButton button[kind="secondary"] {
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        div[data-testid="stChatInput"] {
            background: rgba(15, 23, 42, 0.86);
            border-top: 1px solid rgba(148, 163, 184, 0.16);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PROMPT = PromptTemplate(
    template=(
        "You are a concise assistant answering questions about a YouTube video transcript.\n"
        "Use only the provided context. If the answer is not in the context, say you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer in a short, clear way."
    ),
    input_variables=["context", "question"],
)


def extract_video_id(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Please paste a YouTube link or video ID.")

    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    raise ValueError("Could not extract a valid YouTube video ID.")


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str, language_codes: tuple[str, ...]) -> str:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=list(language_codes))
    return " ".join(snippet.text for snippet in transcript)


@st.cache_resource(show_spinner=False)
def build_vector_store(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
    return FAISS.from_documents(docs, embeddings)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        temperature=0.2,
        api_key=GEMINI_API_KEY,
    )


def format_docs(docs: List) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def count_chunks(text: str) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return len(splitter.split_text(text))


def format_excerpt(text: str, limit: int = 650) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def build_chain(vector_store: FAISS):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    llm = get_llm()
    return parallel | PROMPT | llm | StrOutputParser()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "active_label" not in st.session_state:
    st.session_state.active_label = ""
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.subheader("Start here")
    st.write("Paste a YouTube URL or video ID, choose transcript languages, then load the video once.")
    st.caption("Gemini handles both retrieval embeddings and answers.")

    with st.form("video_loader", clear_on_submit=False):
        source = st.text_input(
            "YouTube link or video ID",
            placeholder="https://www.youtube.com/watch?v=...",
            value=st.session_state.active_label,
        )
        language = st.text_input(
            "Language priority",
            value=st.session_state.selected_language,
            help="Comma-separated codes like en, hi, or en,es.",
        )
        load_button = st.form_submit_button("Load transcript", use_container_width=True)

    if not GEMINI_API_KEY:
        st.warning("Set GOOGLE_API_KEY or GEMINI_API_KEY in a .env file or your environment before chatting.")

    st.caption("Tip: transcript and vector index are cached per video for faster reloads.")

    st.markdown(
        """
        <div class="surface-card" style="margin-top:1rem;">
            <div class="tiny"><strong>Need a quick test?</strong></div>
            <div class="tiny">Load any public video with captions, then ask: “Summarize the main points.”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns([1.3, 0.9], gap="large")

with col1:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">Chat with a YouTube transcript</h1>
            <div class="muted">A simple workspace for loading one video, reviewing the transcript, and asking focused questions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if load_button:
        try:
            video_id = extract_video_id(source)
            language_codes = tuple(code.strip() for code in language.split(",") if code.strip()) or ("en",)
            transcript_text = fetch_transcript(video_id, language_codes)
            vector_store = build_vector_store(transcript_text)
            chunk_count = count_chunks(transcript_text)

            st.session_state.video_id = video_id
            st.session_state.transcript_text = transcript_text
            st.session_state.vector_store = vector_store
            st.session_state.active_label = source
            st.session_state.selected_language = language
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []
            st.success(f"Loaded video {video_id} with transcript ready for chat.")
        except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    if st.session_state.vector_store is None:
        st.markdown(
            """
            <div class="surface-card" style="margin-top:1rem;">
                <h3 style="margin-bottom:0.4rem;">What happens next</h3>
                <div class="tiny">1. Load the transcript from YouTube.</div>
                <div class="tiny">2. Review the transcript preview below.</div>
                <div class="tiny">3. Ask questions in the chat box.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        metrics = st.columns(3)
        metrics[0].metric("Active video", st.session_state.video_id)
        metrics[1].metric("Transcript chars", f"{len(st.session_state.transcript_text):,}")
        metrics[2].metric("Chunks", st.session_state.chunk_count)

        st.markdown(
            """
            <div class="surface-card" style="margin-top:1rem; margin-bottom:0.8rem;">
                <div class="tiny"><strong>Quick prompts</strong></div>
                <div class="tiny">Try: “Summarize the video”, “What are the key takeaways?”, or “List the main steps mentioned.”</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.expander("Show transcript preview", expanded=False):
            st.write(format_excerpt(st.session_state.transcript_text))

        question = st.chat_input("Ask something about the video transcript...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            try:
                chain = build_chain(st.session_state.vector_store)
                answer = chain.invoke(question)
            except Exception as exc:
                answer = f"I could not generate an answer: {exc}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

with col2:
    st.markdown(
        """
        <div class="hero-card">
            <h3 style="margin-bottom:0.5rem;">How it works</h3>
            <div class="tiny">1. Pull transcript from YouTube</div>
            <div class="tiny">2. Split it into searchable chunks</div>
            <div class="tiny">3. Embed chunks with Gemini</div>
            <div class="tiny">4. Retrieve the best matches for each question</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.vector_store is not None:
        st.markdown(
            """
            <div class="hero-card" style="margin-top:1rem;">
                <h3 style="margin-bottom:0.5rem;">Current session</h3>
                <div class="tiny">Transcript is cached for this video.</div>
                <div class="tiny">You can reload the same link with a different language priority.</div>
                <div class="tiny" style="margin-top:0.4rem;">If the answer feels off, try a more specific question or a shorter video.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="hero-card" style="margin-top:1rem;">
            <h3 style="margin-bottom:0.5rem;">Tips</h3>
            <div class="tiny">Use videos with captions for the best results.</div>
            <div class="tiny">Short, specific questions usually work better than broad ones.</div>
            <div class="tiny">Reloading the same video is fast because the transcript is cached.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
