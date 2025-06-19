from flask import Flask, render_template, session, request, jsonify, url_for, g, redirect
import requests, os
from core_system.modules.lotus_model import predict_emotions  # Emotion prediction model GoEmotion
# from core_system.modules.lotus_model_menstrutation import predict_emotions_menstrutation   # Emotion prediction model for menstruation

from core_system.modules.RLLogger import log_interaction, get_emotional_reward # RL-compatible logging system for future RL training
from core_system.utils.kumora_prompt_loader import load_kumora_prompt   # Load system prompt template
from core_system.modules.kumora_response_score.style_score import kumora_style_score, select_kumora_response, select_kumora_emotion_response  # Scoring and selecting responses based on Kumora's emotional style

from dotenv import load_dotenv  
load_dotenv() 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "dev-fallback-secret")
# Local LLM endpoint
LOCAL_TOKENS_ENDPOINT = "http://localhost:1234/v1/chat/completions"
LOCAL_MODEL = "meta-llama-3.1-8b-instruct"

# OpenAI endpoint & token
OPENAI_TOKENS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_TOKEN = os.environ.get("OPENAI_TOKEN")



#print("Check predict_emotions function", predict_emotions("I am happy and excited to travel."))

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
    
# Generate multiple responses with different temperatures for RLAIF Real time emotion feedback
def generate_multiple_responses(messages, temperatures=[0.6, 0.8, 1.0], dominant_emotion=None, emotion_score=0.0):
    responses = []
    for temp in temperatures:
        #print(f"Generating response with temperature: {temp}")
        reply = call_llm(messages, temperature=temp)
        if reply:
            responses.append((reply, temp))
    #print(f"Generated {len(responses)} responses with temperatures {temperatures}")
    #print("Responses:", responses)
    return responses


dialog_history = []

# Your existing questions

questions = {
    "en": [
        "How are you feeling right now? It’s okay if you’re not sure — just say what’s real for you. I am here to listen.",
       
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
        "welcome":  "Welcome beautiful Soul." ,
        "iam_here":  "This space is yours. I'm listening.",
        "you":  "You"
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
    # Clear dialog history only when this route is visited
    session['dialog_history'] = []
    session['emotion_history'] = []

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
    # Analyze message with lotus_model
    analysis = predict_emotions(answer)
    # analysis_menstrutation = predict_emotions_menstrutation(answer)
    # print("Analysis Menstrutation Emotion:", analysis_menstrutation)

    # get (or initialize) this user's history
    history = session.get('dialog_history', [])
    history.append({
        "role": "user", 
        "content": answer,
        "analysis": analysis  # Store analysis with message
    })
    session['dialog_history'] = history
    #print("Dialog History Length:", history)
    # send back next question index and analysis
    return jsonify({
        "status": "ok",
        "analysis": analysis
    })

@app.route('/chat', methods=['POST'])
def chat():
    # session['dialog_history'] =[]  # Reset dialog history for new chat'
    data = request.get_json() or {}
    
    user_prompt = data.get('prompt', '').strip()
    if not user_prompt:
        return jsonify({"error": "Empty user prompt"}), 400

    # Get user language and label info
    user_lang_code = g.user_lang
    language_name = lang_codes.get(user_lang_code, "English")

    # Analyze current message
    emotion_results = data.get('analysis', '')
    dominant_emotion = emotion_results[0][0] if emotion_results and len(emotion_results) > 0 else 'neutral'
    emotion_score = emotion_results[0][1] if emotion_results and len(emotion_results) > 0 else 0.0

    # Load or initialize session history
    dialog_history = session.get('dialog_history', [])
    emotion_history = session.get('emotion_history', [])

    # Append user message with analysis
    user_msg = {
        "role": "user",
        "content": user_prompt,
        "analysis": emotion_results
    }
    dialog_history.append(user_msg)
    emotion_history.append(dominant_emotion)

    # Save session updates
    session['dialog_history'] = dialog_history
    session['emotion_history'] = emotion_history

    # Build emotionally-aware system message
# get select the kumoora response emotion
    select_emo_respo = select_kumora_emotion_response(dominant_emotion)
    kumora_prompt_text = load_kumora_prompt(
        user_prompt=user_prompt,
        dominant_emotion=dominant_emotion,
        emotion_score=emotion_score,
        language_name=language_name,
        emotion_history=", ".join(emotion_history[-3:]),
        select_emo_respo=select_emo_respo
    )

    print("Response " + dominant_emotion, select_emo_respo)
    system_msg = {
        "role": "system",
        "content": kumora_prompt_text
    }

    #print("System Message:", system_msg)
    # Include last few dialog messages for context
    recent_history = dialog_history[-6:]  # up to 3 pairs
    messages = [system_msg] + [
        {"role": turn["role"], "content": turn["content"]} for turn in recent_history
    ]
    #print("Dialog History Length:", len(recent_history))
    # Generate assistant reply
    #reply = call_llm(messages)
    
    # Generate Multiple Responses for RLAIF and pick one with highest emotion score
    responses = generate_multiple_responses(messages, temperatures=[0.6, 0.8, 1.0], dominant_emotion=dominant_emotion, emotion_score=emotion_score)
    print("Generated Responses:", responses)
    reply = select_kumora_response(responses)

    # Save assistant reply
    assistant_msg = {
        "role": "assistant",
        "content": reply
    }
    
    
    dialog_history.append(assistant_msg)
    
    session['dialog_history'] = dialog_history
    
    # ***  RLhf Logging START *** 
    # Analyze the reply for emotions 
    reply_emotion = predict_emotions(reply)
    
    # Get previous emotion from history for reward comparision
    prev_emotion = emotion_history[-2] if len(emotion_history) > 1 else None
    
    
    ## for RLAIF reward calculation
    predicted_emotion_from_reply = reply_emotion[0][0] if reply_emotion else "neutral"
    # Emotion Match Score
    emotion_match_score = 1.0 if predicted_emotion_from_reply == dominant_emotion else 0.0
    # Kumora Style Score
    style_score = kumora_style_score(reply)
    rlaif_reward = round(0.6 * emotion_match_score + 0.4 * style_score, 3)
  
    # Calculate emotional reward based on previous and current emotions for RLHF
    reward = get_emotional_reward(prev_emotion, dominant_emotion)
    
    state = {
        "user_prompt": user_prompt,
        "prev_emotion": prev_emotion,
        "current_emotion": dominant_emotion,
        "emotion_score": emotion_score
    }

    #log_interaction(state, reply, reward)
    log_interaction(state, reply, reward, reply_emotion, rlaif_reward=rlaif_reward)
    
    return jsonify(reply=reply)


@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('welcome'))

if __name__ == '__main__':
    app.run(debug=True)
