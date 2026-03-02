# 🧠 Allen: Data Science Command Center

Allen is an AI-powered personal assistant designed to execute complex data science tasks, manage documents, and hold persistent voice conversations. demo

## 🎯 Project Philosophy: Agents & RAG over Fine-Tuning
A major challenge with modern Large Language Models (LLMs) is that fine-tuning them to learn new information is incredibly resource-intensive. 

Instead of fine-tuning, this project utilizes **Retrieval-Augmented Generation (RAG)** and an **Agentic Framework**. By utilizing intelligent prompting and dynamic context injection, Allen can "learn" from newly uploaded documents and internet searches in real-time. This allows the model to acquire new knowledge instantly without altering its underlying weights, making it highly efficient for constrained hardware environments.

## ✨ Key Features
* **Intelligent Agent:** Powered by `smolagents`, the model understands its persona and autonomously decides which tools to use based on the user's prompt.
* **PDF RAG Engine:** Upload documents and chat with them. The system embeds and retrieves relevant PDF text to answer highly specific questions.
* **Vision & OCR:** Capable of reading text directly from uploaded images or live webcam captures using Tesseract OCR.
* **Persistent Memory:** Integrates a SQLite database to save chat history, allowing the assistant to remember past interactions across multiple sessions.
* **Live Web Search:** Connected to DuckDuckGo, enabling the assistant to pull real-time facts from the internet when its training data falls short.
* **Voice Interactivity:** Features live speech-to-text using `faster-whisper` and realistic text-to-speech utilizing `edge-tts`.

## 🛠️ Tech Stack & Hardware Optimization
This project is optimized to run on a **Google Colab Free Tier (T4 GPU - 15GB VRAM)**. 

* **Core Model:** Qwen 2.5 (7B-Instruct).
* **Quantization:** 8-bit quantization (`BitsAndBytesConfig`) is utilized to reduce the VRAM footprint from ~15GB down to ~8GB, preventing Out of Memory (OOM) errors and leaving space for audio and embedding models.
* **Agent Framework:** SmolAgents & LangChain
* **Vector Database:** FAISS (CPU) with HuggingFace Embeddings
* **Database:** SQLite
* **UI/Interface:** Gradio

## 🚀 Quick Start (Google Colab)

1. Open the `.ipynb` notebook in Google Colab.
2. Ensure your runtime is set to **T4 GPU**.
3. Run the installation cell to set up the 2026-stable AI stack:
   ```bash
   pip install langchain smolagents transformers bitsandbytes accelerate faster-whisper edge-tts gradio pytesseract duckduckgo-search faiss-cpu
   
