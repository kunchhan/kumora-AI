# Kumora: A Soulful AI for Emotional Intelligence and Self-Reflection

<p align="center">
  <img src="https://imgur.com/ihIMBDF.png" alt="Kumora Logo" width="400">
</p>

<p align="center">
  <strong>Kumora begins where traditional AI falls silent—when emotion needs a mirror, not just a solution.</strong>
  <br><br>
  <strong></strong>Beyond Answers: AI with a Heart</strong></p>
  <br><br>
  • <a href="#introduction">Introduction</a><br><br> •
  <a href="#key-features">Key Features</a><br><br> •
  <a href="#system-architecture">System Architecture</a><br><br> •
  <a href="#technology-stack">Tech Stack</a><br><br> •
  <a href="#getting-started">Getting Started</a><br><br> •
  <a href="#project-structure">Project Structure</a><br><br> •
  <a href="#contribution">Contribution</a>
</p>

---

## Introduction

Kumora is an emotionally intelligent AI chatbot designed to provide empathetic, personalized support, particularly for women navigating emotionally significant life moments. While most AI systems excel at logical tasks, they often create an "emotional gap," failing to provide the presence and validation needed for genuine mental wellness support.

This project confronts that challenge by building an AI companion architected not to "fix" problems, but to **gently guide self-reflection**. Inspired by the brain's own self-referential pathways, Kumora listens, remembers, and responds with context-aware empathy, creating a safe space for users to explore their feelings.

## Key Features

*   **Multi-Label Emotion Analysis:** Identifies a spectrum of 27 distinct emotions from user text, moving beyond simple "sad" or "happy" labels.
*   **Persistent User Memory:** Utilizes a Redis-backed system (with an in-memory fallback) to remember conversation history, user goals, and emotional patterns over time.
*   **Dynamic, Empathetic Responses:** Employs a sophisticated prompt engineering pipeline that tailors every response in tone, empathy, and content based on the user's real-time emotional state and past interactions.
*   **Brain-Inspired Conversational Flow:** The system's logic is modeled on the Cortical Midline Structures, guiding a conversation from inward focus (what you feel now) to personal recall (past experiences) and meaning-making (what this means to you).
*   **Modern, Responsive UI:** A clean, serene, and fully responsive web interface built with Flask and modern CSS for a calming user experience.

## System Architecture

Kumora's intelligence is derived from the seamless integration of three core modules:

<p align="center">
  <img src="https://imgur.com/76L9Hsq.png" alt="Kumora System Architecture Diagram" width="800">
</p>

1.  **Emotion Intelligence Module:**
    *   **Function:** Serves as Kumora's perceptual system.
    *   **Implementation:** A fine-tuned DistilBERT model trained for multi-label classification on 27 emotional states. It provides the initial "inward focus" by understanding the user's feelings.

2.  **Context Management System:**
    *   **Function:** Acts as Kumora's long-term memory.
    *   **Implementation:** A robust Redis database stores user profiles, emotional history, and preferences. This enables the "personal recall" stage of a conversation.

3.  **Response Generation Engine:**
    *   **Function:** The synthesis core that generates the final response.
    *   **Implementation:**
        *   Uses a primary/fallback strategy with powerful LLMs (Google Gemini / OpenAI GPT).
        *   A **Dynamic Prompt Engineering** pipeline constructs a highly specific, multi-layered prompt for every turn, ensuring the response is aligned with Kumora's empathetic persona. This module guides the "meaning-making" process.

## Technology Stack

*   **Backend:** Python, Flask (with async support)
*   **Machine Learning:** PyTorch, Hugging Face Transformers, Scikit-learn
*   **Database / Caching:** Redis
*   **LLM APIs:** Google Gemini, OpenAI
*   **Frontend:** HTML5, CSS3, JavaScript

## Getting Started

Follow these steps to set up and run the Kumora application on your local machine.

### 1. Prerequisites

*   Python 3.9+
*   Redis installed and running (or you can rely on the automatic in-memory fallback).

### 2. Clone the Repository

```bash
git clone https://github.com/kunchhan/kumora-AI.git
cd kumora-project
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root of the project by copying the example file.

```bash
cp .env.example .env
```

Now, open the `.env` file and add your secret API keys:

```
# .env

HF_TOKEN="your_hugging_face_token_here"
GOOGLE_API_KEY="your_google_ai_studio_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
```

### 6. Run the Application

With the setup complete, start the Flask development server.

```bash
python app.py
```

The server will start, and the Kumora engine will be initialized. You will see output in your terminal indicating that the server is running.

### 7. Access Kumora

Open your web browser and navigate to:

**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

You should now see the Kumora chat interface and can begin your first conversation!

## Project Structure

The project is organized into modular components for clarity and maintainability.

```
kumora-project/
├── emotion_intelligence_system/
│   └── emotion_classifier.py
├── context_management/
│   ├── context_management_system.py
│   └── kumora_context.py
└── response_generation
│    ├── class_utils.py
│    ├── prompt_engineering_system.py
│    └── prompt_utils.py
├── static/                   # CSS and JavaScript files
│   ├── style.css
│   └── main.js
├── templates/                # Flask HTML templates
│   └── index.html
├── .env                      # Secret API keys (DO NOT COMMIT)
├── .gitignore
├── app.py                    # Main Flask application
├── kumora_chat_terminal.py   # Open Interactive Chat Window in the terminal
├── kumora_response_engine.py # Core engine logic
└── requirements.txt          # Python dependencies
```

## Contribution

This project was developed by a two-person team:

*   **Kunchhan Lama:** Lead Architect & Frontend Engineer
    *   Designed the overall system architecture and the brain-inspired conceptual model.
    *   Developed the responsive HTML/CSS/JS frontend.

*   **Nawaraj Rai:** NLP & Backend Engineer
    *   Trained and evaluated the `Multi-Label Emotion Classifier`.
    *   Designed the `DynamicPromptEngineering` system.
    *   Designed the `KumoraResponseEngine` system for generating proper response using Gemini or GPT models.

We welcome feedback and suggestions for improvement!
