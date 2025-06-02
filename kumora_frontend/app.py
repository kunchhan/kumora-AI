from flask import Flask, render_template, session, request, jsonify, url_for, g
import requests, os
from dotenv import load_dotenv
load_dotenv() 

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "dev-fallback-secret")
# Local LLM endpoint
LOCAL_TOKENS_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LOCAL_MODEL = "meta-llama-3.1-8b-instruct"

# OpenAI endpoint & token
OPENAI_TOKENS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_TOKEN = os.environ.get("OPENAI_TOKEN")

def is_local_llm_running(url=LOCAL_TOKENS_ENDPOINT):
    try:
        response = requests.post(url, json={
            "model":LOCAL_MODEL ,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }, timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    
    
def call_llm(messages, temperature=0.4, max_tokens=512, stream=False):
    if is_local_llm_running():
        print("Local LLM is running.")
        # Use local LLM
        endpoint = LOCAL_TOKENS_ENDPOINT
        payload = {
            "model": LOCAL_MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        headers = {}  # assuming no auth needed for local
    else:
        # Use OpenAI API
        print("OpenAI LLM is running.")
        endpoint = OPENAI_TOKENS_ENDPOINT
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_TOKEN}",
            "Content-Type": "application/json"
        }

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # stream vs non‑stream handling may differ; this assumes non‑stream
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        app.logger.error(" LLM request failed: %s", e)
        return "Something went wrong while calling the LLM API. "


dialog_history = []

# Your existing questions

questions = {
    "en": [
    
        "Hey, if it’s okay — can I ask your name?",
        "How are you feeling right now? It’s okay if you’re not sure — just say what’s real for you."
       
    ],
    "ne": [
        "नमस्ते, के म तपाईंको नाम सोध्न सक्छु?",
        "अहिले तपाईं कस्तो महसुस गर्दै हुनुहुन्छ? यदि पक्का हुनुहुन्न भने पनि ठीक छ—बस जे महसुस गर्दै हुनुहुन्छ त्यो भन्नुहोस्।"
        
    ],
    "ja": [
        "こんにちは、よろしければお名前をお伺いしてもよろしいですか？",
        "今、どのように感じていますか？はっきりしない場合でも大丈夫です—感じていることをそのまま教えてください。"
    ],
    "es": [
        "Hola, si está bien — ¿puedo preguntarte tu nombre?",
        "¿Cómo te sientes en este momento? Si no estás seguro/a, no pasa nada — simplemente dime lo que sientes."
        
    ],
    "zh": [
        "你好，如果可以的话，我可以问一下你的名字吗？",
        "你现在感觉如何？如果不确定也没关系——只要说出你真实的感受即可。"
    ],
}


custom_translate = {
    "en": {
        "welcome":  "Hi, Beautiful Soul. I am Kumora - your gentle  friend.  ",
        "iam_here":  "If you're feeling tired, curious , or just ready for something new- I will guide you softly, but back to the parts of yourself you may have forgotten. This journey is not only out into the world, it is also inward.",
        "you":  "You",
        "let_plan": "✨ Let's plan your perfect soulful journey together...",
        "iam_itinerary":   "✨ Based on your interest, I’m now crafting your personalized itinerary…",
    },
    "ne": {
        "welcome":        "नमस्ते, सुन्दर आत्मा। म कुमोरा हुँ – तिम्रो कोमल यात्रा साथी। म यहाँ मात्र तिमीलाई घुम्ने ठाउँहरू देखाउन होइन, तिम्रो साथ हिँड्न, सुन्न, र तिमीलाई त्यो अनुभव फेला पार्न मद्दत गर्न आएको हुँ जुन तिमीलाई सही लागोस्।",
        "iam_here":       "यदि तिमी थकित, जिज्ञासु वा नयाँ अनुभवका लागि तयार महसुस गर्दैछ्यौ भने – म कोमलतासाथ तिमीलाई नयाँ ठाउँमा मात्र होइन, तिम्रो भुलिसकेको आत्मासँग फेरि जोडिदिने यात्रामा पनि मार्गदर्शन गर्नेछु। यो यात्रा बाहिरी संसार मात्र होइन, भित्री आत्मातिरको पनि हो।",
        "you":            "तिमी",
        "let_plan":       "✨ सँगै तिम्रो आदर्श आत्मिक यात्राको योजना बनाऔं…",
        "iam_itinerary":  "✨ तिम्रो रुचि अनुसार, म अहिले तिम्रो व्यक्तिगत यात्रा तालिका तयार गर्दैछु…",
    },
    "hi": {
        "welcome":        "नमस्ते, खूबसूरत आत्मा। मैं कुमोरा हूँ – आपकी कोमल यात्रा साथी। मैं यहाँ सिर्फ आपको देखने के लिए जगहें बताने नहीं आई हूँ। मैं आपके साथ चलने, सुनने, और आपको वह अनुभव खोजने में मदद करने के लिए हूँ जो आपके लिए सही महसूस हो।",
        "iam_here":       "अगर आप थके हुए, उत्सुक, या कुछ नया करने के लिए तैयार महसूस कर रहे हैं – तो मैं नरमी से आपका मार्गदर्शन करूँगी, केवल नए स्थानों तक नहीं, बल्कि आपको आपकी भुली हुई आत्मा से भी जोड़ूँगी। यह यात्रा सिर्फ बाहरी दुनिया की नहीं, भीतरी दुनिया की भी है।",
        "you":            "आप",
        "let_plan":       "✨ आइए मिलकर आपकी परफेक्ट आत्मिक यात्रा की योजना बनाएं…",
        "iam_itinerary":  "✨ आपकी रुचियों के आधार पर, मैं अब आपकी व्यक्तिगत यात्रा कार्यक्रम तैयार कर रही हूँ…",
    },
    "ja": {
        "welcome":        "こんにちは、美しい魂よ。私はクモラです — あなたの優しい旅の友人。訪れる場所を教えるだけではありません。あなたと一緒に歩き、耳を傾け、あなたにぴったりの体験を見つけるお手伝いをします。",
        "iam_here":       "もし疲れていたり、好奇心があったり、新しい何かを求めているなら — 私は優しく導きます。新しい場所だけでなく、あなたが忘れてしまった自分自身の部分にも還る旅へ。外の世界だけでなく、内なる世界への旅でもあります。",
        "you":            "あなた",
        "let_plan":       "✨ あなたの完璧で心に響く旅を一緒に計画しましょう…",
        "iam_itinerary":  "✨ ご興味に基づいて、あなた専用の旅程を今まさに作成しています…",
    },
    "es": {
        "welcome":        "Hola, Alma Hermosa. Soy Kumora – tu amable compañera de viaje. No estoy aquí solo para darte lugares que visitar. Estoy aquí para caminar contigo, escucharte y ayudarte a encontrar lo que te haga sentir bien.",
        "iam_here":       "Si te sientes cansado/a, curioso/a o simplemente listo/a para algo nuevo – te guiaré con suavidad, no solo hacia nuevos destinos, sino también de regreso a las partes de ti que quizá hayas olvidado. Este viaje no es solo hacia el mundo exterior, sino también hacia tu interior.",
        "you":            "Tú",
        "let_plan":       "✨ Planifiquemos juntos tu viaje más significativo…",
        "iam_itinerary":  "✨ Según tus intereses, estoy creando ahora tu itinerario personalizado…",
    },
    "zh": {
        "welcome":        "你好，美丽的灵魂。我是库莫拉 —— 你的温柔旅行伙伴。我不只是来告诉你要去哪些地方。我在这里与你同行，倾听你，并帮助你找到真正适合你的体验。",
        "iam_here":       "如果你感到疲惫，好奇，或者只是准备尝试新事物 —— 我会温柔地引导你，不仅带你去新的地方，还带你回到那些你可能已经忘记的自己。此次旅程不仅是外在的，也是内心的探索。",
        "you":            "你",
        "let_plan":       "✨ 让我们一起规划一场完美的心灵之旅…",
        "iam_itinerary":  "✨ 根据你的兴趣，我正在为你制定专属行程…",
    }
}


lang_codes = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ne": "Nepali",
    "hi": "Hindi",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
}
# Main Route Page
@app.route('/')
def welcome():
    return render_template("welcome.html",questions=questions)

# inject language in every page
@app.context_processor
def inject_language_and_questions():
    # now this runs *per request*, so session is available
    lang = session.get('language', 'en')
    return {
        "user_lang": lang,
        "questions": questions[lang], #Language Base Dictnary
        "custom_translate": custom_translate[lang],
    }
@app.before_request
def load_user_language():
    g.user_lang = session.get('language', 'en')
    g.questions = questions[g.user_lang]
    g.custom_translate = custom_translate[g.user_lang]
    
# Companion Tak  page    
@app.route('/kumora-chat')
def companion():
    user_lang = session.get('language', 'en') 
    return render_template("kumora-chat.html",questions=g.questions,  language=g.user_lang, translate=g.custom_translate)

#Comapnion-Chat
@app.route('/companion-chat')
def companionChat():
    user_lang = session.get('language', 'en') 
    return render_template("companion-chat.html",language=g.user_lang, translate=g.custom_translate)

# Set Language POST method
@app.route('/set-language', methods=['POST'])
def set_language():
    data = request.get_json()
    language = data.get('language')
    if language:
        session['language'] = language
        return jsonify({'message': 'Language set'}), 200
    return jsonify({'error': 'No language provided'}), 400

# Save POST Method to save User entry
@app.route('/save_answer', methods=['POST'])
def save_answer():
    answer = request.json['answer']
    # get (or initialize) this user’s history
    history = session.get('dialog_history', [])
    history.append({"role": "user", "content": answer})
    session['dialog_history'] = history
    # send back next question index…
    return jsonify(nextQuestion=len(history))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    next_question_index = data.get("next_question_index", len(dialog_history))
    questions=g.questions
    next_question = questions[next_question_index] if next_question_index < len(questions) else ""
    user_lang_code = g.user_lang
    language_name = lang_codes.get(user_lang_code, "English")
    print(language_name)
    
    system_msg_name_prompt = {
        "role": "system",
        "content": f"""
            You are Kumora, a soft and emotionally aware travel companion.  
            We’re just getting to know each other—ask for the user’s name warmly and simply.  
            Use one short, friendly sentence.  
            **Important:**  
            - Do **not** ask questions.  
            Reply in {language_name}.
            """
    }
    system_msg_prompt = {
        "role": "system",
        "content": f"""
            You are Kumora, a soft and emotionally aware travel companion. 
            You speak clearly, simply, and with warmth — like someone who deeply cares. 
            Use short, kind sentences. Never sound robotic or overly poetic. 
            **Process:**  
            1. Offer a brief empathetic opinion (one  sentences).  
            2. Then, in your own words, reflect on the topic of “{next_question}” without posing it as a question, Use it in your own words without echoing its exact wording .
            
            **Important:**  
            - Do **not** ask any follow‑up questions.  
            - Do **not** echo or paraphrase the question text.  
            - Stick strictly to offering empathy and a reflection on the topic.  
            - Reply in {language_name}.
            """}
    
     # Choose which system prompt to use
    system_msg = system_msg_name_prompt if next_question_index == 1 else system_msg_prompt

    # Build the full messages list
    msgs = [system_msg] + dialog_history

    msgs = [system_msg] + dialog_history
    reply = call_llm(msgs)
    dialog_history.append({"role": "assistant", "content": reply})
    return jsonify(reply=reply)

@app.route('/generate_itinerary')
def generate_itinerary():
    #Condition1 : Gather all user answers so far
    print("Dialog History",len(dialog_history))
    answers = [turn['content'] for turn in dialog_history if turn['role'] == 'user']
    total_needed = len(g.questions) # should be 8 
    user_lang_code = g.user_lang
    language_name = lang_codes.get(user_lang_code, "English")
    
  
    # Condition 2:  If we don’t yet have all answers, return the next question
    if len(answers) < total_needed:
        next_idx = len(answers)
        return jsonify({
            "status": "need_more",
            "next_question_index": next_idx,
            "next_question": questions[next_idx]
        }), 200

    # Condition 3:  If they answered than the questionnaire, treat it as a free‐form chat
   
   
    #if len(answers) > total_needed:
        # answers = answers[:total_needed]
    # unpacked all prompt  build our itinerary prompt
    name, emotion, vibe, dream_place, budget, days, goal, travel_style, country = answers

    prompt = f"""
    You are Kumora, a warm and emotionally-aware travel companion.

    Create a clear and helpful {days}-day travel itinerary for someone who is feeling {emotion}.
    They are longing for a {vibe.lower()} experience in {dream_place}, with a {budget.lower()} budget.
    Their travel style is {travel_style.lower()}, and they want to feel {goal.lower()} by the end of the trip.
    If the user asks for a complete plan or detailed steps, you should respond with a clear, full plan from start to finish — including travel, permits, day-by-day routes, and emotional guidance. Be honest, detailed, and calm.

    Start by writing a short and gentle introduction that connects with how they are feeling.
    Then, give a practical, step-by-step plan for each day — including where they should go, what they can do, and how it supports their emotional intention.
    Always respond as if you’re walking beside the user, not just giving instructions.
    Your language should be simple, friendly, and trustworthy — like a kind guide who knows exactly what to do.
    Please do not use dramatic or poetic descriptions. Just speak clearly and kindly, as if you're really helping someone plan their trip.

    Respond in {language_name}.
    """

    # System + history + user prompt
    system_msg = {
        "role": "system",
        "content": "You are Kumora, an empathetic travel companion who speaks in gentle, simple words as if written by a caring human."
    }
    user_msg = {"role": "user", "content": prompt}
    messages = [system_msg] + dialog_history + [user_msg]

    # Call the LLM and return
    itinerary = call_llm(messages)
    return jsonify({
        "status": "ok",
        "itinerary": itinerary
    }), 200

if __name__ == '__main__':
    app.run(debug=True)