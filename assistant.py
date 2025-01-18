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

def is_mobile_related(text: str) -> bool:
    """Check if the query is related to mobile phones or accessories"""
    mobile_keywords = [
        'phone', 'mobile', 'smartphone', 'iphone', 'android', 'samsung', 'display',
        'screen', 'battery', 'camera', 'charging', 'processor', 'memory', 'storage',
        'ram', 'speaker', 'microphone', 'xiaomi', 'oneplus', 'google pixel', 'realme',
        'redmi', 'oppo', 'vivo', 'huawei', 'nothing', 'asus', 'sony', 'motorola',
        'foldable', 'flip', 'accessories', 'charger', 'headphone', 'earphone', 'case',
        'cover', 'adapter', 'power bank'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in mobile_keywords)

# Add new function to filter recommendations against dataset
def filter_recommendations_by_dataset(recommendations: List[str], dataset: pd.DataFrame) -> List[str]:
    """Filter recommendations to only include products from the dataset"""
    if dataset is None or dataset.empty:
        return recommendations
    
    filtered_recommendations = []
    for rec in recommendations:
        # Check if any product name in dataset contains the recommendation
        matches = dataset[dataset['Phone Name'].str.contains(rec, case=False, na=False)]
        if not matches.empty:
            # Use the exact name from dataset
            filtered_recommendations.append(matches.iloc[0]['Phone Name'])
    
    return filtered_recommendations if filtered_recommendations else ["No matching products found in our inventory"]

# Modify the generate_recommendations_with_gemini function
def generate_recommendations_with_gemini(text: str, api_key: str, context: str = None) -> List[str]:
    """Generate recommendations using Google Gemini API."""
    global phone_dataset, spare_parts_dataset
    
    if not is_mobile_related(text):
        return ["No recommendations available - Query not related to mobile phones"]
    
    # Extract brand from query and check availability
    query_words = text.lower().split()
    brands_in_phones = set(phone_dataset['Phone Name'].str.split().str[0].str.lower())
    requested_brand = next((word for word in query_words if word in brands_in_phones), None)
    
    # Check if this is a troubleshooting query
    if is_troubleshooting_query(text):
        # Get relevant spare parts
        if spare_parts_dataset is not None and not spare_parts_dataset.empty:
            part_types = {
                'display': ['display', 'screen'],
                'battery': ['battery', 'charging', 'power'],
                'camera': ['camera', 'lens'],
                'audio': ['headphone', 'speaker', 'sound', 'audio', 'jack'],
                'charging': ['charger', 'charging port', 'usb'],
                'button': ['button', 'power button', 'volume'],
                'sensor': ['fingerprint', 'sensor']
            }
            
            # Determine part type from query
            part_type = None
            for type_name, keywords in part_types.items():
                if any(keyword in text.lower() for keyword in keywords):
                    part_type = type_name.title()
                    break
            
            # Filter parts
            if part_type:
                # Convert both to lowercase for case-insensitive comparison
                mask = spare_parts_dataset['Type'].str.lower() == part_type.lower()
                matching_parts = spare_parts_dataset[mask]
                
                if not matching_parts.empty:
                    parts = []
                    for _, part in matching_parts.iterrows():
                        # If brand mentioned, filter by brand
                        if requested_brand:
                            if requested_brand.lower() in part['Compatible Models'].lower():
                                parts.append(f"{part['Part Name']} - {part['Price Range']} ({part['Availability']})")
                        else:
                            parts.append(f"{part['Part Name']} - {part['Price Range']} ({part['Availability']})")
                    
                    if parts:
                        return parts[:3]
            
            return ["No specific spare parts found for your issue. Please visit a service center."]
        
        return ["Spare parts database not available. Please contact support."]
    
    # For phone recommendations (non-troubleshooting)
    available_phones = phone_dataset['Phone Name'].tolist()
    
    # If specific brand requested but not available
    if query_words and 'suggest' in text.lower() or 'recommendation' in text.lower():
        brand_requested = next((word for word in query_words if word not in ['suggest', 'me', 'some', 'phones', 'recommend']), None)
        if brand_requested and not any(brand_requested.lower() in phone.lower() for phone in available_phones):
            return [
                f"Brand '{brand_requested.title()}' is not available in our inventory.",
                "Please explore our available brands:",
                "Samsung, Apple, Google, OnePlus, Xiaomi, and more."
            ]
    
    # Rest of the recommendation logic
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Create dataset-aware prompt
        prompt = f"""
        Given this user query: '{text}'
        Context: {context if context else 'general recommendation'}
        
        Available phones in our inventory:
        {', '.join(available_phones)}
        
        Strict Rules:
        1. ONLY recommend phones from the provided inventory list
        2. If a specific brand is requested that's not in our inventory, reply with:
           "This brand is not available in our inventory. Here are alternative recommendations:"
        3. Generate exactly 3 recommendations
        
        Format response as:
        If brand not available:
        "This brand is not available in our inventory. Here are alternative recommendations:"
        1. [Exact Phone Name from inventory]
        2. [Exact Phone Name from inventory]
        3. [Exact Phone Name from inventory]

        If brand available or no specific brand:
        1. [Exact Phone Name from inventory]
        2. [Exact Phone Name from inventory]
        3. [Exact Phone Name from inventory]
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            recommendations = []
            lines = response.text.strip().split('\n')
            
            # Check if brand not available message is present
            if "not available in our inventory" in lines[0].lower():
                recommendations.append(lines[0].strip())
                lines = lines[1:]  # Skip the message line
            
            # Process recommendation lines
            for line in lines:
                cleaned_line = line.strip()
                if cleaned_line and cleaned_line[0].isdigit():
                    recommendation = cleaned_line.split('. ', 1)[-1].strip()
                    if recommendation in available_phones:
                        recommendations.append(recommendation)
            
            # Ensure we have alternatives if brand not found
            if len(recommendations) == 1 and "not available" in recommendations[0]:
                # Get similar phones based on price range and features
                similar_phones = phone_dataset.sample(n=min(3, len(phone_dataset)))['Phone Name'].tolist()
                recommendations.extend(similar_phones)
            
            print("Generated recommendations:", recommendations)
            return recommendations if recommendations else ["No matching products found in our inventory"]
            
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return [f"Error generating recommendations: {str(e)}"]

def recommend_products(crm_data, text):
    """Updated to use Gemini for recommendations"""
    if not is_mobile_related(text):
        return ["No recommendations available - Query not related to mobile phones"]
    
    global api_key
    
    # Check for foldable-specific queries
    if 'foldable' in text.lower():
        return generate_recommendations_with_gemini(text, api_key, context='foldable')
    
    # Check for troubleshooting/spare parts queries
    if is_troubleshooting_query(text):
        return generate_recommendations_with_gemini(text, api_key, context='repair')
    
    # If it's a suggestion request, try to understand the context
    if any(term in text.lower() for term in ['suggest', 'recommendation', 'recommend']):
        context = extract_product_context(text)
        if context:
            return generate_recommendations_with_gemini(text, api_key, context=context)
    
    # For general queries
    return generate_recommendations_with_gemini(text, api_key)

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
    if not is_mobile_related(text):
        return ["No questions available - Query not related to mobile phones"]
    
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
    """Updated to use Gemini for search results"""
    global api_key
    
    if not is_mobile_related(query):
        return [{"name": "No results found", "message": "Query not related to mobile phones"}]
    
    # Generate results using Gemini
    results = generate_search_results_with_gemini(query, api_key)
    
    # Format results for display
    formatted_results = []
    for result in results:
        if 'name' in result:
            if 'display' in result and 'specs' in result:
                # Phone result format
                formatted_results.append({
                    'name': result['name'],
                    'display': result['display'],
                    'specs': result['specs'],
                    'os': result.get('recommendation', 'N/A'),
                    'camera': result.get('price', 'N/A')
                })
            else:
                # Spare part or other result format
                formatted_results.append({
                    'name': result['name'],
                    'type': result.get('display', 'N/A'),
                    'compatible': result.get('specs', 'N/A'),
                    'price': result.get('price', 'N/A'),
                    'availability': result.get('availability', 'N/A'),
                    'message': result.get('recommendation', 'N/A')
                })
    
    return formatted_results if formatted_results else [{"name": "No results found", "message": "No matching results"}]

def is_troubleshooting_query(query: str) -> bool:
    troubleshooting_indicators = [
        'issue', 'problem', 'broken', 'not working', 'repair',
        'replace', 'fix', 'damaged', 'trouble', 'help'
    ]
    return any(indicator in query.lower() for indicator in troubleshooting_indicators)

def get_spare_part_recommendations(query: str) -> List[str]:
    """Get spare part recommendations using Gemini"""
    return generate_recommendations_with_gemini(query, api_key, context='spare parts')

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
    
    if not is_mobile_related(text):
        return {
            'sentiment': 'NEUTRAL',
            'score': 0.0,
            'recommendations': ["No recommendations available - Query not related to mobile phones"],
            'questions': ["No questions available - Query not related to mobile phones"]
        }
    
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
    """Get foldable phone recommendations using Gemini"""
    return generate_recommendations_with_gemini(
        "recommend latest foldable phones", 
        api_key, 
        context='foldable'
    )

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
    """Get context-based recommendations using Gemini"""
    query = f"recommend best {context} phones"
    return generate_recommendations_with_gemini(query, api_key, context=context)

def generate_search_results_with_gemini(query: str, api_key: str) -> List[Dict[str, str]]:
    """Generate search results using Google Gemini API"""
    if not is_mobile_related(query):
        return [{"name": "No results found", "message": "Query not related to mobile phones"}]
    
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Create search-specific prompt
        prompt = f"""
        Given this search query: '{query}'
        
        Generate exactly 3 detailed product or spare part results.
        If it's a troubleshooting query, include relevant spare parts and solutions.
        
        Format each result as a JSON-like structure with these fields:
        1. {{
            "name": "[Product/Part Name]",
            "display": "[Display specs if phone, or type if spare part]",
            "specs": "[Full specifications or part details]",
            "price": "[Price range or estimate]",
            "availability": "[Availability status]",
            "recommendation": "[Why this is recommended]"
        }}
        2. {{Similar structure}}
        3. {{Similar structure}}
        
        Keep information accurate and relevant to the query.
        Include only the structured results, no additional text.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            # Process and clean the response
            results = []
            try:
                import json
                # Extract JSON-like structures and parse them
                response_text = response.text.strip()
                # Split the response into individual JSON objects
                json_strings = [s.strip() for s in response_text.split('}') if s.strip()]
                
                for json_str in json_strings:
                    if json_str:
                        # Add closing brace if missing
                        if not json_str.endswith('}'):
                            json_str += '}'
                        # Remove numbering and clean the string
                        json_str = json_str.lstrip('123.). ')
                        try:
                            result = json.loads(json_str)
                            results.append(result)
                        except json.JSONDecodeError:
                            continue
                
                return results if results else [{"name": "No specific results found", "message": "Try being more specific"}]
            except Exception as e:
                print(f"Error parsing search results: {e}")
                return [{"name": "Error", "message": "Failed to parse search results"}]
    
    except Exception as e:
        print(f"Error generating search results: {e}")
        return [{"name": "Error", "message": f"Error generating results: {str(e)}"}]
