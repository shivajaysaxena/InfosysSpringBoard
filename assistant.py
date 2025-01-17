import speech_recognition as sr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Optional, Dict, Tuple
import pandas as pd

# Global variables - All global variables must be declared at the top
api_key = None
phone_dataset = None
spare_parts_dataset = None
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
    # print(f"Vector database initialized with {len(data)} CRM entries.") 


def generate_embeddings(text):
    return embedding_model.encode(text)

def recommend_products(crm_data, text):
    # Check for foldable-specific queries
    if 'foldable' in text.lower():
        return recommend_foldable_phones()
    
    # Check for troubleshooting/spare parts queries
    if is_troubleshooting_query(text):
        parts = get_spare_part_recommendations(text)
        if parts and parts[0] != "Spare parts database not available":
            return parts
    
    # If it's a suggestion request, try to understand the context
    if any(term in text.lower() for term in ['suggest', 'recommendation', 'recommend']):
        context = extract_product_context(text)
        if context:
            return get_contextual_recommendations(context)
    
    # Original recommendation logic for general product queries
    if not crm_data or vector_db is None:
        return ["No recommendations available."]
    
    query_embedding = np.array([generate_embeddings(text)]).astype(np.float32)
    k = min(5, len(crm_data))
    distances, indices = vector_db.search(query_embedding, k)
    recommendations = [crm_data[i]["product"] for i in indices[0] if i < len(crm_data)]
    
    return list(dict.fromkeys(recommendations)) if recommendations else ["No recommendations available."]

def recommend_affordable_phones(text):
    """Recommend phones based on affordability criteria"""
    affordable_phones = [
        "Google Pixel 9",
        "Nothing 2",
        "Poco F6 Pro",
        "Redmi K70 Pro",
        "Realme GT Neo 6"
    ]
    return affordable_phones

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

def load_datasets(phone_path="phone_comparison.csv", parts_path="spare_parts.csv"):
    global spare_parts_dataset
    try:
        phone_dataset = pd.read_csv(phone_path)
        spare_parts_dataset = pd.read_csv(parts_path)
        print("Datasets loaded successfully")
        return phone_dataset
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def load_phone_dataset(file_path="phone_comparison.csv"):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading phone dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

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
    # First check if it's a troubleshooting query
    if is_troubleshooting_query(query):
        return get_spare_part_recommendations(query)
    
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

def is_troubleshooting_query(query: str) -> bool:
    troubleshooting_indicators = [
        'issue', 'problem', 'broken', 'not working', 'repair',
        'replace', 'fix', 'damaged', 'trouble', 'help'
    ]
    return any(indicator in query.lower() for indicator in troubleshooting_indicators)

def get_spare_part_recommendations(query: str) -> List[str]:
    """Get spare part recommendations"""
    global spare_parts_dataset  # Add this line to explicitly use global variable
    
    if spare_parts_dataset is None or spare_parts_dataset.empty:
        try:
            # Try to reload the dataset
            spare_parts_dataset = pd.read_csv("spare_parts.csv")
            print("Successfully reloaded spare parts dataset")
        except Exception as e:
            print(f"Error loading spare parts dataset: {e}")
            return ["Spare parts database not available. Please try again later."]
    
    matched_parts = []
    query_lower = query.lower()
    
    # Extract issue type from query
    issue_types = {
        'battery': ['battery', 'charging', 'power'],
        'display': ['display', 'screen', 'touch'],
        'camera': ['camera', 'photo', 'lens'],
        'audio': ['speaker', 'sound', 'audio']
    }
    
    issue_type = None
    for type_, keywords in issue_types.items():
        if any(keyword in query_lower for keyword in keywords):
            issue_type = type_
            break
    
    if issue_type:
        print(f"Found issue type: {issue_type}")
        
    for _, part in spare_parts_dataset.iterrows():
        if issue_type and str(part['Type']).lower() == issue_type:
            matched_parts.append({
                'name': part['Part Name'],
                'price': part['Price Range'],
                'availability': part['Availability'],
                'compatible': part['Compatible Models']
            })
    
    # Update the format string to be cleaner
    formatted_parts = [
        f"{part['name']} - {part['price']} ({part['availability']}) - Compatible with: {part['compatible']}"
        for part in matched_parts
    ]
    
    if not formatted_parts:
        return ["No specific spare parts found for your issue. Please visit a service center."]
    
    return formatted_parts[:5]

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

def get_conversation_details(text):
    """Get conversation analysis details"""
    global phone_dataset, api_key
    
    if phone_dataset is None or phone_dataset.empty:
        return {
            'sentiment': 'NEUTRAL',
            'score': 0.0,
            'recommendations': ["Dataset not initialized properly"],
            'questions': ["Please check system initialization"]
        }
    
    sentiment, score = analyze_sentiment(text)
    
    # First check if it's a troubleshooting query
    if is_troubleshooting_query(text):
        recommendations = get_spare_part_recommendations(text)
    else:
        # Only generate CRM data for non-troubleshooting queries
        current_crm_data = generate_crm_data([text], phone_dataset)
        initialize_vector_db(current_crm_data)
        recommendations = recommend_products(current_crm_data, text)
    
    dynamic_questions = generate_dynamic_questions(text, api_key)
    return {
        'sentiment': sentiment,
        'score': score,
        'recommendations': recommendations,
        'questions': dynamic_questions
    }

def initialize_assistant(provided_api_key, phone_data_path="phone_comparison.csv", parts_data_path="spare_parts.csv"):
    """Initialize the assistant with required global variables"""
    global api_key, phone_dataset, spare_parts_dataset
    api_key = provided_api_key
    
    try:
        print(f"Loading datasets from: {phone_data_path} and {parts_data_path}")
        phone_dataset = pd.read_csv(phone_data_path)
        spare_parts_dataset = pd.read_csv(parts_data_path)
        if phone_dataset.empty or spare_parts_dataset.empty:
            raise ValueError("One or both datasets are empty")
        print(f"Successfully loaded {len(phone_dataset)} phones and {len(spare_parts_dataset)} spare parts")
        return True
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        phone_dataset = pd.DataFrame()
        spare_parts_dataset = pd.DataFrame()
        return False

def recommend_foldable_phones():
    """Get list of foldable phones from dataset"""
    if phone_dataset is None or phone_dataset.empty:
        return ["No phone data available"]
    
    foldable_phones = phone_dataset[
        phone_dataset['Keywords'].str.contains('foldable|fold|flip', case=False, na=False)
    ]['Phone Name'].tolist()
    
    return foldable_phones if foldable_phones else ["Samsung Galaxy Z Flip 6", "Google Pixel 9 Pro Fold"]

def extract_product_context(text):
    """Extract product context from query"""
    context_keywords = {
        'foldable': ['foldable', 'fold', 'flip'],
        'gaming': ['gaming', 'game'],
        'camera': ['camera', 'photo'],
        'battery': ['battery', 'long lasting'],
        'premium': ['premium', 'flagship']
    }
    
    text_lower = text.lower()
    for context, keywords in context_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return context
    return None

def get_contextual_recommendations(context):
    """Get recommendations based on context"""
    if phone_dataset is None or phone_dataset.empty:
        return ["No phone data available"]
    
    context_filters = {
        'foldable': lambda df: df[df['Keywords'].str.contains('foldable|fold|flip', case=False, na=False)],
        'gaming': lambda df: df[df['Keywords'].str.contains('gaming|performance', case=False, na=False)],
        'camera': lambda df: df[df['Keywords'].str.contains('camera|photo', case=False, na=False)],
        'battery': lambda df: df[df['Battery Capacity'].str.contains('5000|5500|6000', case=False, na=False)],
        'premium': lambda df: df[df['Keywords'].str.contains('premium|flagship', case=False, na=False)]
    }
    
    if context in context_filters:
        filtered_df = context_filters[context](phone_dataset)
        return filtered_df['Phone Name'].tolist()[:5]
    return []
