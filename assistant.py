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
    """Updated to handle feature-based recommendations"""
    if not is_mobile_related(text):
        return ["No recommendations available - Query not related to mobile phones"]
    
    global api_key
    
    # Extract features from query
    features = extract_features_from_query(text)
    
    # If features found, use feature-based recommendations
    if features:
        return generate_feature_based_recommendations(text, api_key, features)
    
    # Otherwise use existing recommendation logic
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
    """Updated to handle issues better"""
    global api_key
    
    if not is_mobile_related(query):
        return [{"name": "No results found", "message": "Query not related to mobile phones"}]
    
    # Check if this is an issue-related query
    if is_troubleshooting_query(query):
        return generate_issue_results_with_gemini(query, api_key)
    
    # For non-issue queries, use existing search logic
    return generate_search_results_with_gemini(query, api_key)

def is_troubleshooting_query(query: str) -> bool:
    troubleshooting_indicators = [
        'issue', 'problem', 'broken', 'not working', 'repair',
        'replace', 'fix', 'damaged', 'trouble', 'help'
    ]
    return any(indicator in query.lower() for indicator in troubleshooting_indicators)

def get_spare_part_recommendations(query: str) -> List[str]:
    """Get spare part recommendations using Gemini"""
    return generate_recommendations_with_gemini(query, api_key, context='spare parts')

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
    
    # Configure Gemini API right after setting api_key
    genai.configure(api_key=provided_api_key)
    print("Configured Gemini API with provided key")
    
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
    
    global phone_dataset
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Create search-specific prompt with actual phone data
        prompt = f"""
        Given this search query: '{query}'
        And this phone data:
        {phone_dataset[['Phone Name', 'Display', 'RAM', 'Storage', 'Battery Capacity', 'Processor', 'Keywords']].to_string()}
        
        Generate exactly 3 detailed product results based on the phone data.
        
        Format each result as a complete JSON object with these exact fields:
        {{
            "name": "EXACT Phone Name from dataset",
            "display": "Full Display specs from dataset",
            "specs": "Detailed specs including RAM, Storage, Battery, Processor",
            "price_category": "Budget($200-400)/Mid($400-800)/Premium($800+)",
            "keywords": "Key features and selling points"
        }}

        Ensure each response is proper JSON format and use actual data from the dataset.
        """
        
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                import json
                # Clean up the response text to ensure valid JSON
                text = response.text.strip()
                # Extract all JSON objects
                import re
                json_objects = re.findall(r'\{[^{}]*\}', text)
                
                results = []
                for json_str in json_objects:
                    try:
                        result = json.loads(json_str)
                        # Verify required fields
                        if all(key in result for key in ['name', 'display', 'specs', 'price_category']):
                            results.append(result)
                    except json.JSONDecodeError:
                        continue
                
                return results if results else [{"name": "No specific results found", "display": "N/A", "specs": "N/A", "price_category": "N/A"}]
                
            except Exception as e:
                print(f"Error parsing search results: {e}")
                return [{"name": "Error", "display": "N/A", "specs": "Error parsing results", "price_category": "N/A"}]
    
    except Exception as e:
        print(f"Error generating search results: {e}")
        return [{"name": "Error", "display": "N/A", "specs": f"Error: {str(e)}", "price_category": "N/A"}]

def generate_issue_results_with_gemini(query: str, api_key: str) -> List[Dict[str, str]]:
    """Generate troubleshooting results using Google Gemini API"""
    try:
        # Add fallback responses for common issues
        issue_types = {
            'display': {
                'name': 'Display Issue',
                'type': 'Hardware Issue',
                'diagnosis': 'Common display problems include screen flickering, dead pixels, touch responsiveness issues, or physical damage.',
                'solution': 'Basic troubleshooting steps:\n1. Restart device\n2. Check for software updates\n3. Test in safe mode\n4. Check physical damage\n5. Visit authorized service center if issue persists',
                'parts': 'May require display assembly replacement',
                'cost': 'Varies by model, typically $200-500',
                'warning': 'Please backup your data before any repairs'
            },
            'battery': {
                'name': 'Battery Issue',
                'type': 'Hardware Issue',
                'diagnosis': 'Common battery problems include rapid drain, not charging, or swelling.',
                'solution': 'Basic troubleshooting steps:\n1. Check charging cable and adapter\n2. Clear cache\n3. Check battery-heavy apps\n4. Consider battery replacement if old',
                'parts': 'Battery pack replacement may be needed',
                'cost': 'Typically $30-100 depending on model',
                'warning': 'Use only genuine batteries from authorized sources'
            },
            'general': {
                'name': 'General Issue',
                'type': 'Hardware/Software',
                'diagnosis': 'Requires specific diagnosis based on symptoms',
                'solution': 'Basic troubleshooting steps:\n1. Restart device\n2. Check for updates\n3. Clear cache\n4. Factory reset if needed',
                'parts': 'To be determined after diagnosis',
                'cost': 'Varies based on issue',
                'warning': 'Always backup data before major repairs'
            }
        }

        # Determine issue type from query
        issue_type = 'general'
        for key in issue_types.keys():
            if key in query.lower():
                issue_type = key
                break

        # Use fallback response if Gemini API fails
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze this mobile phone issue: '{query}'
        Provide detailed troubleshooting steps.
        Format response as JSON with fields:
        - name: issue title
        - type: Hardware/Software
        - diagnosis: problem description
        - solution: step-by-step fix
        - parts: required parts
        - cost: estimated cost
        - warning: safety notes
        """
        
        try:
            response = model.generate_content(prompt, timeout=5)  # Add timeout
            if response and response.text:
                import json
                results = json.loads(response.text)
                if isinstance(results, dict):
                    return [results]
        except Exception as api_error:
            print(f"Gemini API error: {api_error}")
            # Fall back to pre-defined response
            return [issue_types[issue_type]]
            
        # Fall back to pre-defined response if parsing fails
        return [issue_types[issue_type]]
        
    except Exception as e:
        print(f"Error in generate_issue_results: {e}")
        return [issue_types['general']]

def extract_features_from_query(text: str) -> Dict[str, List[str]]:
    """Extract phone features from query text"""
    feature_keywords = {
        'display': ['amoled', 'oled', 'lcd', 'screen', 'display', 'inch', '"', '120hz', '90hz', 'refresh'],
        'camera': ['camera', 'mp', 'megapixel', 'ultra', 'wide', 'zoom', 'photo', 'selfie', 'rear'],
        'performance': ['processor', 'snapdragon', 'dimensity', 'gaming', 'fps', 'ram', 'storage', 'gb'],
        'battery': ['mah', 'battery', 'charging', 'fast charge', 'power'],
        'special': ['foldable', 'flip', 'waterproof', 'rugged', '5g', 'wireless']
    }
    
    found_features = {category: [] for category in feature_keywords}
    text_lower = text.lower()
    
    for category, keywords in feature_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_features[category].append(keyword)
    
    return {k: v for k, v in found_features.items() if v}

def generate_feature_based_recommendations(text: str, api_key: str, features: Dict[str, List[str]]) -> List[str]:
    """Generate recommendations based on specific features using Gemini"""
    global phone_dataset
    
    if not features:
        return ["No specific features mentioned in the query"]
    
    feature_summary = ", ".join([f"{k}: {', '.join(v)}" for k, v in features.items()])
    available_phones = phone_dataset['Phone Name'].tolist()
    
    prompt = f"""
    Find phones matching these features: {feature_summary}
    From this list of available phones: {', '.join(available_phones)}
    
    Rules:
    1. ONLY recommend phones from the provided list
    2. MUST match at least one requested feature
    3. Recommend exactly 3 phones if possible
    4. Format as:
       1. [Phone Name] - [Why it matches the requirements]
       2. [Phone Name] - [Why it matches the requirements]
       3. [Phone Name] - [Why it matches the requirements]
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        if response.text:
            recommendations = []
            for line in response.text.strip().split('\n'):
                if line.strip() and line[0].isdigit():
                    phone_name = line.split('-')[0].split('.')[1].strip()
                    if phone_name in available_phones:
                        recommendations.append(phone_name)
            
            return recommendations if recommendations else ["No phones match the specified features"]
    except Exception as e:
        return [f"Error generating recommendations: {str(e)}"]
