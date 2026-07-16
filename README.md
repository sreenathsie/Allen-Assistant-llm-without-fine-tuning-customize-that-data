# Allen — Personal AI Assistant (RAG + Agents, No Fine-Tuning)

An agentic personal assistant built on top of **Qwen2.5-7B-Instruct**, running entirely on a free Google Colab T4 GPU. Instead of fine-tuning the model on new information, Allen uses **Retrieval-Augmented Generation (RAG)** and an **agent framework** to pull in documents, images, and live web results at query time — so it can "learn" new information instantly, without ever touching the model's weights.

## Project philosophy: agents & RAG over fine-tuning

Fine-tuning an LLM to learn new information is expensive: it requires GPU time, curated training data, and has to be repeated every time the underlying knowledge changes. This project takes a different approach — instead of baking knowledge into the model, Allen retrieves relevant information on demand (from uploaded PDFs, images, or the web) and injects it into the prompt as context. The model itself never changes; only what it's shown at inference time changes. This makes the system flexible and cheap to update (add a new PDF, get new knowledge, instantly) at the cost of relying entirely on prompt quality and retrieval accuracy rather than learned knowledge.

## What this project does

Allen is a chat assistant with:
- **Text and voice conversation** — type or speak, get spoken replies back
- **Document Q&A** — upload a PDF and ask questions about its specific contents
- **Image/photo text reading** — OCR on uploaded images or live webcam captures
- **Live web search** — pulls current information from DuckDuckGo when the question needs it
- **Persistent memory** — remembers past conversations across sessions via a SQLite database stored on Google Drive
- **Toggleable tools** — search, PDF retrieval, and OCR can each be switched on/off per-conversation from the UI

It runs as a Gradio web app inside Colab, with a shareable public link.

## Real-world use cases

- Personal research assistant: upload papers/reports, ask specific questions, get answers grounded in that document rather than the model's general knowledge
- Quick OCR + Q&A on photographed text (whiteboards, printed pages, screenshots)
- Voice-driven assistant for hands-free interaction
- A practical template for building a RAG+agent system without needing to fine-tune anything — useful as a reference implementation for anyone wanting to combine retrieval, tools, and an LLM

**Not suitable for**: tasks requiring guaranteed factual accuracy without a source document (the model can still hallucinate outside of RAG-provided context), production/multi-user deployment (single Gradio session, SQLite has no concurrency handling), or any use requiring low latency (7B model + agent loop + tool calls is slow on a single T4).

## Architecture & tech stack

| Component | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Strong instruction-following at a size that (with quantization) fits a free-tier GPU |
| Quantization | 8-bit (`BitsAndBytesConfig`) | Cuts VRAM footprint from ~15GB to ~8GB, leaving headroom for embeddings/audio models on a 15GB T4 |
| Agent framework | `smolagents` (`CodeAgent`) | Lets the model decide which tool to call based on the prompt, rather than hardcoding logic |
| RAG / vector store | FAISS (CPU) + HuggingFace embeddings (`all-MiniLM-L6-v2`) | Lightweight, no GPU needed for retrieval, keeps GPU free for the LLM |
| PDF parsing | LangChain `PyPDFLoader` | Splits PDFs into chunks for embedding and retrieval |
| OCR | Tesseract via `pytesseract` | Extracts text from uploaded images and webcam captures |
| Speech-to-text | `faster-whisper` | Local, GPU-accelerated transcription, no external API |
| Text-to-speech | `edge-tts` | Free, natural-sounding voices, no API key required |
| Memory | SQLite, stored on Google Drive | Persists chat history across Colab sessions/reconnects |
| UI | Gradio (`Blocks`) | Fast to build a multi-input (text/mic/file/camera) interface, shareable via public link |

## How it works

1. **Model loading** (Cell 1): Qwen2.5-7B-Instruct is downloaded once to Google Drive, then loaded in 8-bit via `smolagents.TransformersModel` with `BitsAndBytesConfig(load_in_8bit=True)`.
2. **Tool setup**: three tools are defined — `search_pdf` (queries the FAISS vector store built from uploaded PDFs), `read_image` (runs OCR via Tesseract), and DuckDuckGo web search (built into `smolagents`).
3. **Per-message flow** (`chat_engine`):
   - Transcribes any microphone input via `faster-whisper`
   - Processes uploaded files: PDFs are chunked and embedded into the FAISS store; images are flagged for OCR
   - Builds the active tool list based on which toggles (Search / PDF RAG / OCR) are enabled in the UI
   - Constructs a prompt including recent conversation context and any file/OCR notes
   - Runs a `CodeAgent` (from `smolagents`) with the selected tools and up to 5 reasoning steps
   - Saves the exchange to SQLite and generates a spoken reply via `edge-tts`
4. **UI**: a single Gradio `Blocks` app with text input, microphone, file upload, webcam capture, per-tool toggles, voice/model selection dropdowns, and buttons to clear the visible chat or wipe persistent memory entirely.

## Advantages

- No training/fine-tuning cost or time — set up and running in one Colab session
- Knowledge updates instantly (upload a new PDF, it's queryable immediately) with no retraining
- Runs entirely on free-tier hardware via 8-bit quantization
- Multi-modal: text, voice, images, and documents in one interface
- Persistent memory survives Colab disconnects (stored on Drive, not in Colab's ephemeral runtime)
- Modular tool toggles — each capability (search/RAG/OCR) can be independently enabled or disabled per message

## Disadvantages / limitations

- **Not a knowledge-updating model** — the base model's own knowledge is frozen; RAG only adds what's explicitly retrieved into the prompt, so questions outside both the model's training data and the retrieved context can still be answered incorrectly (hallucination risk remains for anything not grounded in a source)
- **Single-user, single-session design** — SQLite has no real concurrency support; not built for multiple simultaneous users
- **Latency** — a 7B model with an agent reasoning loop (up to 5 steps) plus tool calls (search, retrieval, OCR) is noticeably slower than a direct model query; not suited for low-latency use
- **Hardcoded paths** — the model path (`/content/drive/MyDrive/1llm/Qn`) and memory DB path are specific to the original author's Drive; anyone reusing this needs to adjust these paths and download the model themselves first
- **Free Colab constraints** — session timeouts and disconnects apply the same as any Colab free-tier project; long conversations or large PDF uploads may need session babysitting
- **`use_think` toggle behavior**: when disabled, the assistant currently just returns a fixed string ("Deep Think is off...") instead of a lighter-weight response — this is a placeholder behavior worth revisiting rather than true "fast mode"
- **Error handling is broad**: tool/agent failures are caught with a generic exception handler and surfaced as a plain error string to the user, which is fine for a personal project but not descriptive enough for debugging at scale

## Future goals

- Replace hardcoded Drive paths with a config file or environment variables for portability
- Give the "Think" toggle real behavior (e.g. skip the agent loop and hit the model directly) instead of returning a placeholder message when off
- Add support for more document types beyond PDF (e.g. .docx, .txt, .csv) to the RAG pipeline
- Add proper concurrency/session handling if moving beyond single-user personal use
- More structured logging/error surfacing for easier debugging of agent tool-call failures
- Consider a lighter-weight/distilled model option for lower latency, with the 7B model as an optional "high quality" mode

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
└── Llm_Personal_assistant_without_finetune.ipynb
```

## Quick start (Google Colab)

1. Open the `.ipynb` notebook in Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run the model-download cell (Cell 1) — this saves Qwen2.5-7B-Instruct to your Google Drive at `/content/drive/MyDrive/1llm/Qn` (adjust the path in the notebook if you want it elsewhere)
4. Run the dependency install cell, or install directly:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the model-loading cell, then the main app cell — a Gradio interface will launch with a public shareable link

**Note:** the notebook's own dependency cell installs `gradio>=4.0` (the app code relies on Gradio 4's `sources=` API for webcam input) — make sure you're not pinned to an older Gradio version.

## Acknowledgements

- [Qwen2.5](https://huggingface.co/Qwen) — Alibaba Cloud
- [smolagents](https://github.com/huggingface/smolagents) — Hugging Face
- [LangChain](https://www.langchain.com/) for RAG components
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [edge-tts](https://github.com/rany2/edge-tts) for voice I/O
