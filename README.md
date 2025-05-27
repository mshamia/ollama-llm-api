# 🧠 Ollama Chat & Prompt API

This is a Flask-based API that provides endpoints to interact with [Ollama](https://ollama.com/) models. It includes:

- Chatting with LLMs (via `/chat`)
- Enhancing prompts for image/video generation (via `/enhance-prompt`)
- Listing, pulling, and tracking models
- Service health checks

---

## 🚀 Features

- ✅ Chat with LLM models like `llama3.2`
- 🎨 Enhance vague prompts into rich, detailed image/video generation prompts
- 🔍 List all available Ollama models
- ⬇️ Pull models on-demand
- 📊 Track model download status
- ❤️ Health check endpoint

---

## 📦 Requirements

- Python 3.8+
- `ollama` Python client
- Flask

Install dependencies:

```bash
pip install flask requests ollama
