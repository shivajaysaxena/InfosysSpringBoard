# import speech_recognition as sr
# from transformers import pipeline

# # Sentiment analysis pipeline using Hugging Face
# sentiment_analyzer = pipeline("sentiment-analysis")

# def recognize_speech(file_path):
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(file_path) as source:
#             audio = recognizer.record(source)
#         recognized_text = recognizer.recognize_google(audio)
#         return recognized_text
#     except sr.UnknownValueError:
#         return "Sorry, I couldn't understand the speech."
#     except sr.RequestError as e:
#         return f"Error with the speech recognition service: {e}"
#     except Exception as e:
#         return f"An unexpected error occurred: {e}"

# def analyze_sentiment(text):
#     try:
#         result = sentiment_analyzer(text)
#         sentiment = result[0]['label']
#         score = result[0]['score']
#         return sentiment, score
#     except Exception as e:
#         return f"Sentiment analysis failed: {e}", 0.0





# m3
# import speech_recognition as sr
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss

# # Load models
# sentiment_analyzer = pipeline("sentiment-analysis")
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# vector_db = None

# def recognize_speech(file_path):
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(file_path) as source:
#             audio = recognizer.record(source)
#         recognized_text = recognizer.recognize_google(audio)
#         return recognized_text
#     except sr.UnknownValueError:
#         return "Sorry, I couldn't understand the speech."
#     except sr.RequestError as e:
#         return f"Error with the speech recognition service: {e}"
#     except Exception as e:
#         return f"An unexpected error occurred: {e}"

# def initialize_vector_db(data):
#     global vector_db
#     if not data:
#         print("No valid CRM data available. Skipping vector database initialization.")
#         return

#     embeddings = [embedding_model.encode(entry['text']) for entry in data]
#     vector_db = faiss.IndexFlatL2(len(embeddings[0]))
#     vector_db.add(np.array(embeddings))
#     print("Vector database initialized with CRM data.")

# def generate_embeddings(text):
#     return embedding_model.encode(text)

# def recommend_products(crm_data, text):
#     if not crm_data:
#         return ["No recommendations available."]

#     query_embedding = generate_embeddings(text)
#     distances, indices = vector_db.search(np.array([query_embedding]), k=3)
#     recommendations = [crm_data[i]['product'] for i in indices[0] if i < len(crm_data)]
#     return recommendations

# def generate_dynamic_questions(text):
#     return [
#         f"Can you elaborate on '{text}'?",
#         f"What is the main concern regarding '{text}'?",
#         "How can we address your current needs better?"
#     ]

# def analyze_sentiment(text):
#     try:
#         result = sentiment_analyzer(text)
#         sentiment = result[0]['label']
#         score = result[0]['score']
#         return sentiment, score
#     except Exception as e:
#         return f"Sentiment analysis failed: {e}", 0.0

# def generate_crm_data(conversations):
#     products = [
#         {"keywords": ["phone", "smartphone"], "product": "Smartphone A"},
#         {"keywords": ["laptop", "affordable"], "product": "Laptop B"},
#         {"keywords": ["fitness", "tracker"], "product": "Fitness Band C"}
#     ]

#     crm_data = []
#     for conversation in conversations:
#         for product in products:
#             if any(keyword in conversation.lower() for keyword in product["keywords"]):
#                 crm_data.append({"text": conversation, "product": product["product"]})

#     return crm_data


# import speech_recognition as sr
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss

# # Load models
# sentiment_analyzer = pipeline("sentiment-analysis")
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# vector_db = None

# # Load Open-Source LLM 
# llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.3")
# llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-3.3")

# # Speech Recognition
# def recognize_speech(file_path):
#     recognizer = sr.Recognizer()
#     try:
#         with sr.AudioFile(file_path) as source:
#             audio = recognizer.record(source)
#         recognized_text = recognizer.recognize_google(audio)
#         return recognized_text
#     except sr.UnknownValueError:
#         return "Sorry, I couldn't understand the speech."
#     except sr.RequestError as e:
#         return f"Error with the speech recognition service: {e}"
#     except Exception as e:
#         return f"An unexpected error occurred: {e}"

# # Initialize Vector Database
# def initialize_vector_db(data):
#     global vector_db
#     if not data:
#         print("No valid CRM data available. Skipping vector database initialization.")
#         return

#     embeddings = [embedding_model.encode(entry['text']) for entry in data]
#     vector_db = faiss.IndexFlatL2(len(embeddings[0]))
#     vector_db.add(np.array(embeddings))
#     print("Vector database initialized with CRM data.")

# # Generate Embeddings
# def generate_embeddings(text):
#     return embedding_model.encode(text)

# # Product Recommendations
# def recommend_products(crm_data, text):
#     if not crm_data or vector_db is None:
#         return ["No recommendations available."]

#     query_embedding = generate_embeddings(text)
#     distances, indices = vector_db.search(np.array([query_embedding]), k=3)
#     recommendations = [crm_data[i]['product'] for i in indices[0] if i < len(crm_data)]
#     return recommendations or ["No suitable products found."]

# # Generate Dynamic Questions with LLM
# def generate_dynamic_questions(text):
#     prompt = f"Generate three questions to better understand the user's concern based on: '{text}'"
#     inputs = llm_tokenizer(prompt, return_tensors="pt")
#     outputs = llm_model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
#     questions = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return questions.split("\n")[:3]  # Return top 3 questions

# # Sentiment Analysis
# def analyze_sentiment(text):
#     try:
#         result = sentiment_analyzer(text)
#         sentiment = result[0]['label']
#         score = result[0]['score']
#         return sentiment, score
#     except Exception as e:
#         return f"Sentiment analysis failed: {e}", 0.0

# # Generate CRM Data
# def generate_crm_data(conversations):
#     products = [
#         {"keywords": ["phone", "smartphone"], "product": "Smartphone A"},
#         {"keywords": ["laptop", "affordable"], "product": "Laptop B"},
#         {"keywords": ["fitness", "tracker"], "product": "Fitness Band C"},
#         {"keywords": ["battery", "backup"], "product": "Power Bank D"}
#     ]

#     crm_data = []
#     for conversation in conversations:
#         for product in products:
#             if any(keyword in conversation.lower() for keyword in product["keywords"]):
#                 crm_data.append({"text": conversation, "product": product["product"]})

#     return crm_data





import logging
logging.basicConfig(level=logging.WARNING)  # Adjust as needed: INFO, ERROR
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')
from sentence_transformers import SentenceTransformer
import speech_recognition as sr

# Set up logging for better error tracking and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = None

# from transformers import AutoTokenizer, AutoModelForCausalLM
# Replace with a publicly accessible model name
# model_name = "EleutherAI/gpt-neo-2.7B"  # Example
# llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
# llm_model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", cache_dir="./models")

# Speech Recognition
def recognize_speech(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        recognized_text = recognizer.recognize_google(audio)
        logger.info(f"Recognized Speech: {recognized_text}")
        return recognized_text
    except sr.UnknownValueError:
        logger.error("Speech could not be understood.")
        return "Sorry, I couldn't understand the speech."
    except sr.RequestError as e:
        logger.error(f"Error with speech recognition service: {e}")
        return f"Error with the speech recognition service: {e}"
    except Exception as e:
        logger.exception("An unexpected error occurred during speech recognition.")
        return f"An unexpected error occurred: {e}"

# Initialize Vector Database
def initialize_vector_db(data):
    global vector_db
    if not data:
        logger.warning("No valid CRM data available. Skipping vector database initialization.")
        return

    embeddings = [embedding_model.encode(entry['text']) for entry in data]
    vector_db = faiss.IndexFlatL2(len(embeddings[0]))
    vector_db.add(np.array(embeddings))
    logger.info("Vector database initialized with CRM data.")

# Generate Embeddings
def generate_embeddings(text):
    return embedding_model.encode(text)

# Product Recommendations
def recommend_products(crm_data, text):
    if not crm_data or vector_db is None:
        logger.warning("CRM data or vector database not available. Returning default recommendation.")
        return ["No recommendations available."]

    query_embedding = generate_embeddings(text)
    distances, indices = vector_db.search(np.array([query_embedding]), k=3)
    recommendations = []
    for i in indices[0]:
        if i < len(crm_data):
            recommendations.append(crm_data[i]['product'])
        else:
            recommendations.append("No suitable products found.")
    return recommendations

# Generate Dynamic Questions with LLM
def generate_dynamic_questions(text):
    prompt = f"Generate three questions to better understand the user's concern based on: '{text}'"
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        outputs = llm_model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
        questions = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return questions.split("\n")[:3]  # Return top 3 questions
    except Exception as e:
        logger.exception("Error generating dynamic questions with LLM.")
        return ["Error generating questions."]

# Sentiment Analysis
def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        sentiment = result[0]['label']
        score = result[0]['score']
        logger.info(f"Sentiment: {sentiment}, Score: {score}")
        return sentiment, score
    except Exception as e:
        logger.exception("Sentiment analysis failed.")
        return f"Sentiment analysis failed: {e}", 0.0

# Generate CRM Data
def generate_crm_data(conversations):
    products = [
        {"keywords": ["phone", "smartphone"], "product": "Smartphone A"},
        {"keywords": ["laptop", "affordable"], "product": "Laptop B"},
        {"keywords": ["fitness", "tracker"], "product": "Fitness Band C"},
        {"keywords": ["battery", "backup"], "product": "Power Bank D"}
    ]

    crm_data = []
    for conversation in conversations:
        for product in products:
            if any(keyword in conversation.lower() for keyword in product["keywords"]):
                crm_data.append({"text": conversation, "product": product["product"]})

    logger.info(f"Generated CRM Data: {len(crm_data)} entries")
    return crm_data


