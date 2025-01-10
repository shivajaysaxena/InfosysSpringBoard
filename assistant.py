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
# import google.generativeai as genai
# from typing import List, Optional
# import pandas as pd

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
#     print("Trying to find matching products...")
#     if not crm_data:
#         return ["No recommendations available."]

#     query_embedding = generate_embeddings(text)
#     distances, indices = vector_db.search(np.array([query_embedding]), k=3)
#     recommendations = [crm_data[i]["product"] for i in indices[0] if i < len(crm_data)]
#     return recommendations

# def generate_dynamic_questions(text: str, api_key: str) -> List[str]:
#     """
#     Generate dynamic questions using Google Gemini API.
    
#     Args:
#         text (str): Input text to generate questions for
#         api_key (str): Google API key for authentication
        
#     Returns:
#         List[str]: List of generated questions. Returns error messages if generation fails.
#     """
#     # Configure the Generative AI library
#     genai.configure(api_key=api_key)
    
#     try:
#         # Initialize the model
#         model = genai.GenerativeModel('gemini-pro')
        
#         # Create the prompt
#         prompt = f"""
#         Given this text: '{text}'
        
#         Generate exactly 3 thoughtful questions based on this content.
#         Generate only questions realted to products and nothing else.
#         Format your response as a numbered list with just the questions, like this:
#         1. [First question]
#         2. [Second question]
#         3. [Third question]
        
#         Do not include any other text or explanations.
#         """
        
#         # Generate the response
#         response = model.generate_content(prompt)
        
#         # Extract questions from the response
#         if response.text:
#             # Split the response into lines and clean them
#             questions = []
#             for line in response.text.strip().split('\n'):
#                 # Remove numbering and whitespace
#                 cleaned_line = line.strip()
#                 if cleaned_line:
#                     # Remove the number prefix (e.g., "1. ", "2. ")
#                     question = cleaned_line.split('. ', 1)[-1].strip()
#                     questions.append(question)
#             print("questons:", questions)
#             return questions if questions else ["No questions were generated."]
            
#     except Exception as e:
#         print(f"Error occurred while generating questions: {str(e)}")
#         return [f"Error generating questions: {str(e)}"]

# def analyze_sentiment(text):
#     try:
#         result = sentiment_analyzer(text)
#         sentiment = result[0]['label']
#         score = result[0]['score']
#         return sentiment, score
#     except Exception as e:
#         return f"Sentiment analysis failed: {e}", 0.0

# def load_phone_dataset(file_path="phone_comparison.csv"):
#     try:
#         phones = pd.read_csv(file_path)
#         return phones
#     except Exception as e:
#         print(f"Error loading phone dataset: {e}")
#         return None

# # Generate CRM data dynamically based on conversations and phone dataset
# def generate_crm_data(conversations, phone_dataset):
#     if phone_dataset is None or phone_dataset.empty:
#         print("Phone dataset is empty or not available.")
#         return []

#     crm_data = []
#     for conversation in conversations:
#         print(f"Processing conversation: {conversation}")
#         for _, row in phone_dataset.iterrows():
#             keywords = row["Keywords"].split(",")  # Assuming a "Keywords" column in the dataset
#             # print(f"Checking product: {row['Phone Name']}")
#             # print(f"Keywords: {keywords}")
#             if any(keyword.strip().lower() in conversation.lower() for keyword in keywords) or any(term in conversation.lower() for term in ["mobile", "phone", "smartphone"]):
#                 crm_data.append({"text": conversation, "product": row["Phone Name"]})
#     print("CRM data generated.")
#     return crm_data

























import speech_recognition as sr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Optional
import pandas as pd

# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = None

def recognize_speech(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        recognized_text = recognizer.recognize_google(audio)
        return recognized_text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the speech."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def initialize_vector_db(data):
    global vector_db
    if not data:
        print("No valid CRM data available. Skipping vector database initialization.")
        return

    # Create embeddings for all CRM data
    embeddings = np.array([embedding_model.encode(entry['text']) for entry in data]).astype(np.float32)
    
    # Initialize the Faiss vector database with the correct dimensionality
    vector_db = faiss.IndexFlatL2(embeddings.shape[1])  # Number of dimensions of embeddings
    vector_db.add(embeddings)
    print(f"Vector database initialized with {len(data)} CRM entries.") 


def generate_embeddings(text):
    return embedding_model.encode(text)

def recommend_products(crm_data, text):
    print("Processing conversation:", text)
    
    if not crm_data or vector_db is None:
        return ["No recommendations available."]
    
    # Embed the query
    query_embedding = np.array([generate_embeddings(text)]).astype(np.float32)
    
    # Perform the search using Faiss
    distances, indices = vector_db.search(query_embedding, k=5)  # k=5 to diversify results
    
    # Get recommended products based on the indices
    recommendations = [crm_data[i]["product"] for i in indices[0] if i < len(crm_data)]
    
    if not recommendations:
        return ["No recommendations available."]
    
    return recommendations


def generate_dynamic_questions(text: str, api_key: str) -> List[str]:
    """
    Generate dynamic questions using Google Gemini API.
    
    Args:
        text (str): Input text to generate questions for
        api_key (str): Google API key for authentication
        
    Returns:
        List[str]: List of generated questions. Returns error messages if generation fails.
    """
    # Configure the Generative AI library
    genai.configure(api_key=api_key)
    
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Create the prompt
        prompt = f"""
        Given this text: '{text}'
        
        Generate exactly 3 thoughtful questions based on this content.
        Generate only questions related to products and nothing else.
        Format your response as a numbered list with just the questions, like this:
        1. [First question]
        2. [Second question]
        3. [Third question]
        
        Do not include any other text or explanations.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Extract questions from the response
        if response.text:
            # Split the response into lines and clean them
            questions = []
            for line in response.text.strip().split('\n'):
                cleaned_line = line.strip()
                if cleaned_line:
                    question = cleaned_line.split('. ', 1)[-1].strip()
                    questions.append(question)
            print("questions:", questions)
            return questions if questions else ["No questions were generated."]
            
    except Exception as e:
        print(f"Error occurred while generating questions: {str(e)}")
        return [f"Error generating questions: {str(e)}"]

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        sentiment = result[0]['label']
        score = result[0]['score']
        return sentiment, score
    except Exception as e:
        return f"Sentiment analysis failed: {e}", 0.0

def load_phone_dataset(file_path="phone_comparison.csv"):
    try:
        phones = pd.read_csv(file_path)
        return phones
    except Exception as e:
        print(f"Error loading phone dataset: {e}")
        return None

# Generate CRM data dynamically based on conversations and phone dataset
def generate_crm_data(conversations, phone_dataset):
    if phone_dataset is None or phone_dataset.empty:
        print("Phone dataset is empty or not available.")
        return []

    crm_data = []
    for conversation in conversations:
        print(f"Processing conversation: {conversation}")
        for _, row in phone_dataset.iterrows():
            keywords = row["Keywords"].split(",")  # Assuming a "Keywords" column in the dataset
            if any(keyword.strip().lower() in conversation.lower() for keyword in keywords) or any(term in conversation.lower() for term in ["mobile", "phone", "smartphone"]):
                crm_data.append({"text": conversation, "product": row["Phone Name"]})
    print("CRM data generated.")
    return crm_data


