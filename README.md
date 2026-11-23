# YouTube RAG Chat

Minimal Streamlit app for chatting with a YouTube video transcript.

## What this project does

This app lets a user paste a YouTube URL or video ID, fetch the transcript, split it into chunks, embed those chunks with Gemini, store them in FAISS, and chat against the most relevant transcript context.

The current app uses:

- Streamlit for the UI
- `youtube-transcript-api` for transcript extraction
- Gemini for embeddings and chat
- FAISS for vector retrieval
- LangChain to wire the retrieval and generation pipeline together

## What was done

The original notebook-based workflow was turned into a public-friendly Streamlit application with a cleaner interface, a transcript preview, a sidebar loader, cached transcript/vector indexing, and a Gemini-backed retrieval flow.

The app was also updated so it does not rely on a hardcoded API key. It now reads credentials from environment variables or a local `.env` file.

## How it works

1. A YouTube link or video ID is entered in the sidebar.
2. The app fetches the transcript with `youtube-transcript-api`.
3. The transcript is split into overlapping chunks.
4. The chunks are embedded with Gemini Embedding 2.
5. FAISS retrieves the most relevant chunks for each question.
6. Gemini 3.6 Flash answers using only the retrieved transcript context.

## File structure to commit to GitHub

Keep these files in the repository:

- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [README.md](README.md)
- [.gitignore](.gitignore)
- [.env.example](.env.example)



## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Gemini key:

```env
GOOGLE_API_KEY=your_key_here
```

Optional overrides:

```env
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

You can also set `GEMINI_API_KEY` instead of `GOOGLE_API_KEY`.

### 5. Run the app

```bash
streamlit run app.py
```
### for mac 
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

rm -rf qdrant_storage
streamlit run streamlit_app.py

## Good default models

Current supported defaults used by the app:

- Chat: `gemini-3.6-flash`
- Embeddings: `gemini-embedding-2`

## Usage

1. Paste a YouTube link or video ID.
2. Set transcript language priority if needed, for example `en` or `en,hi`.
3. Click **Load transcript**.
4. Ask questions in the chat box.

## Troubleshooting

- If the app says there is no API key, confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set.
- If transcript loading fails, the video may not have captions or transcript access may be disabled.
- If a model is retired, change `GEMINI_MODEL` or `GEMINI_EMBEDDING_MODEL` in `.env` without editing code.

## Notes

- The app caches transcript fetches and vector stores per video to keep repeat loads fast.
- The old exploratory notebook is not included in the GitHub-ready file set because the Streamlit app is now the runtime entry point.
