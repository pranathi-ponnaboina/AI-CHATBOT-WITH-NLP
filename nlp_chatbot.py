"""
=============================================================
  AI CHATBOT WITH NLP — CODTECH INTERNSHIP PROJECT
  Built with: NLTK + spaCy + TF-IDF similarity
=============================================================
"""

import re
import sys
import random
import string
import warnings
warnings.filterwarnings("ignore")

# ── Install dependencies if missing ──────────────────────────
try:
    import nltk
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    import spacy
    nlp = spacy.load("en_core_web_sm")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

# ── NLTK downloads ────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
STOPWORDS  = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "howdy", "greetings", "good morning",
                     "good afternoon", "good evening", "sup", "whats up"],
        "responses": [
            "Hello! 👋 I'm your AI assistant. How can I help you today?",
            "Hi there! Great to see you. What's on your mind?",
            "Hey! I'm ready to help. What would you like to know?",
            "Greetings! How can I assist you today?",
        ]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "farewell", "quit", "exit",
                     "take care", "see ya", "later", "cya"],
        "responses": [
            "Goodbye! Have a wonderful day! 🌟",
            "See you later! Feel free to come back anytime. 👋",
            "Farewell! It was great chatting with you!",
            "Take care! Hope I was helpful! 😊",
        ]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thank", "thx", "ty", "appreciate",
                     "grateful", "cheers"],
        "responses": [
            "You're welcome! Happy to help! 😊",
            "Anytime! That's what I'm here for.",
            "Glad I could help! Let me know if you need anything else.",
            "My pleasure! Don't hesitate to ask more questions.",
        ]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "what are you called",
                     "your name", "tell me your name", "introduce yourself"],
        "responses": [
            "I'm NLPBot 🤖 — an AI chatbot built with NLTK and spaCy for CODTECH!",
            "My name is NLPBot! I was built using Natural Language Processing libraries.",
            "I'm NLPBot, your AI assistant powered by NLTK, spaCy, and TF-IDF! 🚀",
        ]
    },
    "capabilities": {
        "patterns": ["what can you do", "your capabilities", "how can you help",
                     "what do you know", "features", "abilities"],
        "responses": [
            "I can answer questions on topics like Python, AI/ML, NLP, math, science, "
            "general knowledge, and more! I use TF-IDF similarity and intent matching. 🧠",
            "I'm trained on a knowledge base covering tech, science, and general topics. "
            "I understand natural language using NLTK and spaCy! Try asking me anything.",
        ]
    },
    "python": {
        "patterns": ["what is python", "tell me about python", "python programming",
                     "python language", "why use python", "python features"],
        "responses": [
            "Python 🐍 is a high-level, interpreted programming language known for its "
            "simplicity and readability. It's widely used in web development, data science, "
            "AI/ML, automation, and more. Created by Guido van Rossum in 1991.",
            "Python is one of the world's most popular languages! Its clean syntax, "
            "huge ecosystem (NumPy, Pandas, TensorFlow), and versatility make it a top "
            "choice for beginners and experts alike.",
        ]
    },
    "nlp": {
        "patterns": ["what is nlp", "natural language processing", "explain nlp",
                     "how does nlp work", "nlp applications", "nltk", "spacy"],
        "responses": [
            "Natural Language Processing (NLP) 🗣️ is a branch of AI that enables computers "
            "to understand, interpret, and generate human language. Libraries like NLTK and "
            "spaCy make NLP in Python straightforward.",
            "NLP bridges human language and computers! It powers chatbots, translation, "
            "sentiment analysis, and search engines. NLTK provides tools like tokenization "
            "and lemmatization; spaCy offers fast, industrial-strength NLP pipelines.",
        ]
    },
    "machine_learning": {
        "patterns": ["what is machine learning", "machine learning", "explain ml",
                     "deep learning", "neural networks", "ai vs ml"],
        "responses": [
            "Machine Learning (ML) 🤖 is a subset of AI where systems learn from data "
            "to improve their performance without being explicitly programmed. Types include "
            "supervised, unsupervised, and reinforcement learning.",
            "ML algorithms include Linear Regression, Decision Trees, Random Forests, "
            "SVMs, and Neural Networks. Deep Learning uses multi-layer neural networks "
            "to solve complex tasks like image recognition and language generation.",
        ]
    },
    "artificial_intelligence": {
        "patterns": ["what is ai", "artificial intelligence", "explain ai",
                     "history of ai", "ai applications"],
        "responses": [
            "Artificial Intelligence (AI) 🧠 is the simulation of human intelligence by "
            "machines. It encompasses ML, NLP, computer vision, robotics, and more. "
            "AI is transforming healthcare, finance, education, and countless industries.",
            "AI was coined as a field in 1956 by John McCarthy. Modern AI includes "
            "narrow AI (task-specific) and the pursuit of AGI (general intelligence). "
            "Applications range from self-driving cars to ChatGPT!",
        ]
    },
    "data_science": {
        "patterns": ["what is data science", "data science", "data analyst",
                     "big data", "data engineering"],
        "responses": [
            "Data Science 📊 combines statistics, programming, and domain expertise to "
            "extract insights from data. A data scientist collects, cleans, analyzes, "
            "and visualizes data to drive business decisions.",
            "The data science toolkit includes Python, R, SQL, Pandas, NumPy, Matplotlib, "
            "Scikit-learn, and TensorFlow. It overlaps with ML, AI, and analytics.",
        ]
    },
    "internet": {
        "patterns": ["what is the internet", "how does internet work", "explain internet",
                     "world wide web", "www"],
        "responses": [
            "The Internet 🌐 is a global network of interconnected computers that communicate "
            "using standardized protocols (TCP/IP). The World Wide Web (WWW) is a service "
            "built on top of the Internet, invented by Tim Berners-Lee in 1989.",
            "The Internet works by breaking data into packets, routing them across networks, "
            "and reassembling them at the destination. DNS translates domain names to IP "
            "addresses, making the web navigable.",
        ]
    },
    "climate_change": {
        "patterns": ["what is climate change", "global warming", "climate crisis",
                     "greenhouse effect", "carbon emissions"],
        "responses": [
            "Climate change 🌍 refers to long-term shifts in global temperatures and weather "
            "patterns, primarily driven by human activities like burning fossil fuels. "
            "It causes rising sea levels, extreme weather, and ecosystem disruption.",
            "The greenhouse effect occurs when CO₂ and other gases trap heat in Earth's "
            "atmosphere. Since the Industrial Revolution, global temperatures have risen "
            "~1.2°C. Limiting warming to 1.5°C is the Paris Agreement goal.",
        ]
    },
    "math": {
        "patterns": ["what is calculus", "explain algebra", "what is mathematics",
                     "pythagorean theorem", "what is pi", "explain statistics"],
        "responses": [
            "Mathematics 📐 is the science of numbers, shapes, and patterns. Key branches "
            "include Algebra (equations), Calculus (change & motion), Geometry (shapes), "
            "Statistics (data analysis), and Number Theory.",
            "Some fun math facts: π (pi) ≈ 3.14159 is the ratio of a circle's circumference "
            "to its diameter. The Pythagorean theorem states a² + b² = c² for right triangles. "
            "Euler's identity e^(iπ) + 1 = 0 is called the most beautiful equation!",
        ]
    },
    "jokes": {
        "patterns": ["tell me a joke", "joke", "make me laugh", "funny", "humor"],
        "responses": [
            "Why do programmers prefer dark mode? 🌙 Because light attracts bugs! 🐛",
            "Why did the AI break up with the algorithm? It had too many unresolved issues! 😄",
            "How many programmers does it take to change a light bulb? None — that's a hardware problem! 💡",
            "Why did the data scientist get lost in the forest? Too many trees in the decision tree! 🌳",
        ]
    },
    "time": {
        "patterns": ["what time is it", "current time", "what is the date", "today's date"],
        "responses": [
            lambda: f"I don't have real-time access, but you can check the time on your device! ⏰",
        ]
    },
    "weather": {
        "patterns": ["what is the weather", "weather today", "will it rain",
                     "temperature today"],
        "responses": [
            "I don't have real-time weather data. Please check weather.com or your local "
            "weather app for current conditions! 🌤️",
        ]
    },
    "health": {
        "patterns": ["what is health", "how to be healthy", "healthy lifestyle",
                     "exercise tips", "diet advice", "mental health"],
        "responses": [
            "Good health 💪 involves balanced nutrition, regular exercise (150+ min/week), "
            "adequate sleep (7-9 hrs), stress management, and regular check-ups. "
            "Small consistent habits make the biggest difference!",
            "Mental health is as important as physical health! Practice mindfulness, "
            "maintain social connections, get enough sleep, and don't hesitate to seek "
            "professional support when needed. 🧘",
        ]
    },
    "default": {
        "responses": [
            "That's an interesting question! I'm not sure about that yet. Could you rephrase or ask something else?",
            "Hmm, I don't have enough information on that. Try asking about Python, AI, NLP, science, or math!",
            "I'm still learning! That topic is outside my current knowledge base. 🤔",
            "Great question! Unfortunately I don't have a confident answer for that yet.",
        ]
    }
}

# ─────────────────────────────────────────────────────────────
#  NLP UTILITIES
# ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, lemmatize, remove stopwords."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    doc = nlp(text)
    tokens = [
        lemmatizer.lemmatize(token.text)
        for token in doc
        if token.text not in STOPWORDS and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


def extract_entities(text: str) -> list:
    """Extract named entities using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def detect_intent_by_pattern(user_input: str) -> str | None:
    """Match intent by checking keywords/patterns directly."""
    lowered = user_input.lower()
    for intent, data in KNOWLEDGE_BASE.items():
        if intent == "default":
            continue
        patterns = data.get("patterns", [])
        for pattern in patterns:
            if pattern in lowered:
                return intent
    return None


def detect_intent_by_tfidf(user_input: str) -> str | None:
    """Use TF-IDF cosine similarity as a fallback intent detector."""
    all_patterns = []
    all_intents  = []
    for intent, data in KNOWLEDGE_BASE.items():
        if intent == "default":
            continue
        for pattern in data.get("patterns", []):
            all_patterns.append(preprocess(pattern))
            all_intents.append(intent)

    if not all_patterns:
        return None

    processed_input = preprocess(user_input)
    corpus = all_patterns + [processed_input]

    try:
        tfidf  = TfidfVectorizer()
        matrix = tfidf.fit_transform(corpus)
        sims   = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        best   = int(np.argmax(sims))
        if sims[best] > 0.25:            # confidence threshold
            return all_intents[best]
    except Exception:
        pass

    return None


def get_response(intent: str) -> str:
    """Pick a random response for the matched intent."""
    data = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["default"])
    responses = data.get("responses", KNOWLEDGE_BASE["default"]["responses"])
    chosen = random.choice(responses)
    return chosen() if callable(chosen) else chosen


def chat(user_input: str) -> dict:
    """
    Main chat function.
    Returns a dict with: response, intent, entities, confidence
    """
    if not user_input.strip():
        return {"response": "Please type something! I'm listening. 👂",
                "intent": "empty", "entities": [], "confidence": 0.0}

    # Entity extraction
    entities = extract_entities(user_input)

    # Intent detection (pattern → TF-IDF → default)
    intent = detect_intent_by_pattern(user_input)
    confidence = 1.0 if intent else 0.0

    if not intent:
        intent = detect_intent_by_tfidf(user_input)
        confidence = 0.7 if intent else 0.0

    if not intent:
        intent = "default"
        confidence = 0.0

    response = get_response(intent)

    return {
        "response":   response,
        "intent":     intent,
        "entities":   entities,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────
#  CLI INTERFACE
# ─────────────────────────────────────────────────────────────

def main():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           🤖  NLPBot — CODTECH AI Chatbot Project            ║
║        Built with NLTK · spaCy · TF-IDF · scikit-learn      ║
╠══════════════════════════════════════════════════════════════╣
║  Type your message and press ENTER to chat.                  ║
║  Type 'quit', 'exit', or 'bye' to end the session.           ║
║  Type '/debug' to toggle debug info (intent + entities).     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    debug_mode = False

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNLPBot: Goodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "/debug":
            debug_mode = not debug_mode
            print(f"[Debug mode {'ON' if debug_mode else 'OFF'}]\n")
            continue

        result = chat(user_input)

        print(f"\nNLPBot: {result['response']}\n")

        if debug_mode:
            print(f"  ┌─ Intent    : {result['intent']}")
            print(f"  ├─ Confidence: {result['confidence']:.2f}")
            if result["entities"]:
                ents = ", ".join(f"{t} [{l}]" for t, l in result["entities"])
                print(f"  └─ Entities  : {ents}")
            else:
                print("  └─ Entities  : none detected")
            print()

        # Exit on farewell intents
        if result["intent"] == "farewell":
            break


if __name__ == "__main__":
    main()
