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
from typing import List, Optional, Dict, Tuple
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


def process_object_query(query: str, phone_dataset: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Process object-based queries to find relevant phones based on use case.
    
    Args:
        query (str): Search query from user
        phone_dataset (pd.DataFrame): Dataset containing phone information
    
    Returns:
        List[Dict[str, str]]: List of relevant phones with their details
    """
    # Define category keywords
    categories = {
        'gaming': ['gaming', 'game', 'pubg', 'fps', 'performance', 'powerful'],
        'camera': ['camera', 'photo', 'photography', 'selfie', 'videography'],
        'business': ['business', 'work', 'professional', 'productivity'],
        'display': ['display', 'screen', 'view', 'watching', 'multimedia'],
        'battery': ['battery', 'long lasting', 'endurance', 'backup']
    }
    
    # Identify category from query
    query_lower = query.lower()
    matched_categories = []
    
    for category, keywords in categories.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_categories.append(category)
    
    # Also check if any keywords from the dataset match
    keyword_matches = phone_dataset[phone_dataset['Keywords'].str.contains(query_lower, case=False, na=False)]
    
    results = []
    if matched_categories:
        for category in matched_categories:
            filtered_phones = filter_phones_by_category(category, phone_dataset)
            results.extend(filtered_phones)
    elif not keyword_matches.empty:
        # If we have keyword matches but no category matches
        results = [create_phone_dict(row) for _, row in keyword_matches.iterrows()]
    else:
        # If no specific category or keyword is matched, use semantic search
        results = semantic_phone_search(query, phone_dataset)
    
    # Remove duplicates based on phone name
    unique_results = []
    seen_phones = set()
    for phone in results:
        if phone['name'] not in seen_phones:
            unique_results.append(phone)
            seen_phones.add(phone['name'])
    
    return unique_results[:5]  # Return top 5 unique results

def filter_phones_by_category(category: str, phone_dataset: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Filter phones based on category-specific criteria.
    """
    def extract_number(value):
        try:
            return pd.to_numeric(''.join(filter(str.isdigit, str(value))))
        except:
            return 0

    criteria = {
        'gaming': lambda df: df[
            (df['RAM'].apply(extract_number) >= 6) &  # At least 6GB RAM
            (df['Processor'].str.contains('Snapdragon|Dimensity|Gaming', case=False, na=False))
        ],
        'camera': lambda df: df[
            (df['Rear Camera'].str.contains('108MP|64MP|48MP', case=False, na=False)) |
            (df['Front Camera'].str.contains('32MP|20MP', case=False, na=False))
        ],
        'business': lambda df: df[
            (df['RAM'].apply(extract_number) >= 6) &
            (df['Storage'].apply(extract_number) >= 128)
        ],
        'display': lambda df: df[
            df['Resolution'].str.contains('2K|1440|AMOLED|120Hz', case=False, na=False)
        ],
        'battery': lambda df: df[
            df['Battery Capacity'].apply(extract_number) >= 5000
        ]
    }
    
    try:
        filtered_df = criteria.get(category, lambda df: df)(phone_dataset)
        return [create_phone_dict(row) for _, row in filtered_df.iterrows()]
    except Exception as e:
        print(f"Error filtering phones for category {category}: {e}")
        return []

def create_phone_dict(row: pd.Series) -> Dict[str, str]:
    """
    Create a standardized dictionary from a phone dataset row.
    """
    return {
        'name': str(row['Phone Name']),
        'display': str(row['Display']),
        'camera': f"Front: {str(row['Front Camera'])}, Rear: {str(row['Rear Camera'])}",
        'specs': f"RAM: {str(row['RAM'])}, Storage: {str(row['Storage'])}, Battery: {str(row['Battery Capacity'])}, Processor: {str(row['Processor'])}",
        'os': str(row['OS'])
    }

def semantic_phone_search(query: str, phone_dataset: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Perform semantic search on phone dataset using query embeddings.
    """
    try:
        # Create query embedding
        query_embedding = embedding_model.encode(query)
        
        # Create embeddings for phone descriptions
        phone_descriptions = phone_dataset.apply(
            lambda row: f"{row['Phone Name']} {row['Keywords']} {row['Display']} {row['Processor']}", 
            axis=1
        ).tolist()
        
        phone_embeddings = np.array([embedding_model.encode(desc) for desc in phone_descriptions])
        
        # Calculate similarities
        similarities = np.dot(phone_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-5:][::-1]  # Get top 5 matches
        
        return [create_phone_dict(phone_dataset.iloc[idx]) for idx in top_indices]
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []
