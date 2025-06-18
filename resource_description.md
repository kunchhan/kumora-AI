### **1. Datasets**

This section details the data sources used to train and validate the core machine learning models.

*   **Primary Training Dataset:** A custom-curated, hybrid dataset created specifically for this project.
    *   **Composition:**
        *   **GoEmotions:** A large-scale public dataset of Reddit comments used for its conversational breadth. We filtered and mapped its labels to our custom 27-emotion schema.
            *   *Source:* [Google Research GitHub](https://github.com/google-research/google-research/tree/master/goemotions)
        *   **ISEAR (International Survey on Emotion Antecedents and Reactions):** A public dataset containing structured self-reports of emotional situations, used for high-quality examples of core emotions.
            *   *Source:* [University of Geneva CISA](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
        *   **Scraped Web Data:** Manually annotated public comments scraped from domain-specific communities to ensure model relevance.
            *   *Platforms:* YouTube, Reddit
            *   *Topics:* Women's wellness, mental health, and personal growth sub-forums.
        *   **Synthetic Data:** Text generated artificially to augment and balance under-represented emotion classes in our training set.

### **2. Pre-trained Models and AI Services**

This section lists the foundational models and external AI services leveraged in the project.

*   **Emotion Classifier Base Model:**
    *   **Model:** `distilbert-base-uncased`
    *   **Provider:** Hugging Face
    *   **Purpose:** Served as the starting point for our fine-tuned multi-label emotion classifier due to its excellent balance of performance and efficiency.
    *   *Source:* [Hugging Face Model Hub](https://huggingface.co/distilbert-base-uncased)

*   **Large Language Models (LLMs) for Response Generation:**
    *   **Primary Model:** `gemini-1.5-flash-latest`
    *   **Provider:** Google AI
    *   **Purpose:** The main engine for generating Kumora's conversational responses, chosen for its speed, quality, and conversational tuning.
    *   *Source:* [Google AI for Developers](https://ai.google.dev/)

    *   **Fallback Model:** `gpt-4.1-mini`
    *   **Provider:** OpenAI
    *   **Purpose:** Used as a resilient fallback to ensure high availability of the chat service in case of primary model failure.
    *   *Source:* [OpenAI API Platform](https://platform.openai.com/)

### **3. Software, Libraries, and Frameworks**

This section details the key software packages that form the technical backbone of the project. A complete list is available in the `requirements.txt` file.

*   **Backend Framework:** Flask (v2.0+ with `[async]` support)
*   **Machine Learning & NLP:** PyTorch, Hugging Face Transformers, Scikit-learn, Pandas
*   **Database & Caching:** Redis
*   **API Interaction:** `openai` Python client, `google-generativeai` Python client
*   **Development & Environment:** Python 3.9+, pip with virtual environments (`venv`)

### **4. Cloud Services and Infrastructure**

This section lists any external cloud services used for hosting, computation, or storage.

*   **API Keys & Secrets Management:** Handled locally via a `.env` file using the `python-dotenv` library.
*   **[Optional] Model Training Environment:**
    *   The emotion classifier model was trained using Google Colab to leverage its complimentary GPU (NVIDIA T4) resources.
    *   Rest of the inference, evaluation, and development were done on a normal local machine.

### **5. Design and Frontend Assets**

This section lists resources used for the user interface and branding.

*   **Logo and Branding:** The Kumora logo and brand identity were custom-designed for this project.
*   **Iconography:**
    *   **Library:** Font Awesome (v6.5.1)
    *   **Purpose:** Provided all UI icons (new chat, send, paperclip, etc.) for a clean and universally understood interface.
    *   *Source:* [Font Awesome CDN](https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css)
*   **Fonts:**
    *   **Library:** Google Fonts
    *   **Fonts Used:** 'Lora' (for branding/headings) and 'Nunito Sans' (for body/chat text).
    *   *Source:* [Google Fonts API](https://fonts.google.com/)