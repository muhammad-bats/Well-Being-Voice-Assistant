# Well-Being Voice Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange)

An end-to-end web application delivering real-time, AI-driven well-being support via voice interaction. This system processes raw audio input, converts it to text, and routes queries to either a high-performance commercial API or a domain-specifically fine-tuned local LLM for therapeutic responses.

## System Architecture

The application utilizes a Flask-based backend to bridge the browser's MediaRecorder API with downstream NLP pipelines. 

- **Frontend:** HTML/JS/CSS capturing microphone audio and handling asynchronous HTTP requests.
- **Speech-to-Text (STT):** `annyang` processing raw audio buffers and mappoing onto JavaScript.
- **Inference Engine Routing:**
  - **Pipeline A (Production/General):** Cohere API integration for low-latency, generalized zero-shot generation.
  - **Pipeline B (Specialized/Local):** A custom-tuned GPT-2 architecture operating locally via HuggingFace `transformers` and PyTorch.
- **Fine-Tuning Corpus:** The local model was aggressively fine-tuned on the `CounselChat` dataset, adapting its weights to mimic the tone, structure, and pacing of licensed professional therapists.

## Tech Stack
- **Backend Core:** Python 3.10, Flask, Werkzeug
- **Machine Learning & NLP:** PyTorch, HuggingFace Transformers, Tokenizers
- **External APIs:** Cohere AI
- **Frontend Delivery:** Vanilla JavaScript, HTML5 Web Audio API, CSS3

## Model Fine-Tuning Pipeline
The local inference model was produced using the `fine_tuning_bats.py` pipeline.
- **Base Architecture:** `gpt2` (124M parameters)
- **Dataset:** `CounselChat` (High-quality Q&A paired with licensed therapist responses)
- **Hyperparameters:** Batch Size: 8, Epochs: 3
- **Generation Tuning:** The inference script (`app.py`) utilizes repetition penalties to constrain outputs to concise, professional advice.

## Technical Challenges & Lessons Learned
- **Audio Latency & Buffer Management:** Bridging the gap between a browser's asynchronous audio stream and a synchronous Python backend required optimizing the payload size and selecting a robust STT intermediate step to prevent blocking the main thread.
- **Domain Adaptation vs. Overfitting:** Fine-tuning GPT-2 on `CounselChat` presented the challenge of catastrophic forgetting. The hyperparameter search was carefully constrained to ensure the model adopted a therapeutic tone without losing its baseline conversational coherence.
- **API Fallback Strategy:** Integrating Cohere alongside a local model demonstrated the architectural trade-offs between API latency/cost and local compute constraints/memory bottlenecks.
