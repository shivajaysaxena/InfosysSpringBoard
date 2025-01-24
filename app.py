from dotenv import load_dotenv
import os
import sqlite3
import streamlit as st
import speech_recognition as sr
from datetime import datetime
from assistant import (
    analyze_sentiment,
    recommend_products,
    generate_dynamic_questions,
    generate_crm_data,
    initialize_vector_db,
    load_phone_dataset,
    process_object_query,
    is_troubleshooting_query,
    initialize_assistant
)
from fpdf import FPDF
import time
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import google.generativeai as genai
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Access the API key and validate
api_key = os.getenv('API_KEY')
if not api_key:
    st.error("API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Initialize assistant with validated API key
success = initialize_assistant(api_key, "phone_comparison.csv", "spare_parts.csv")
if not success:
    st.error("Failed to initialize assistant. Please check your datasets and API key.")
    st.stop()
print("Assistant initialized successfully")

# Initialize database path
database_path = "conversations.db"

# Initialize the database
def initialize_database():
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Save conversation to the database
def save_conversation(session_id, text):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (session_id, text) VALUES (?, ?)", (session_id, text))
    conn.commit()
    conn.close()

# Fetch past conversations for the session
def fetch_past_conversations(session_id):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT text, timestamp 
        FROM conversations 
        WHERE session_id = ? 
        ORDER BY timestamp DESC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Add new function to get conversation details
def get_conversation_details(text):
    sentiment, score = analyze_sentiment(text)
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

# Generate PDF
def generate_pdf(transcripts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Speech Analysis Report", ln=True, align='C')

    for entry in transcripts:
        pdf.ln(10)
        pdf.multi_cell(0, 10,
            f"Transcript: {entry['text']}\n"
            f"Sentiment: {entry['sentiment']}\n"
            f"Score: {entry['score']}\n"
            f"Shift: {entry['shift']}\n"
            f"Recommendations: {', '.join(entry['recommendations'])}\n"
            f"Questions: {', '.join(entry['questions'])} "
        )

    pdf_output = "transcripts_report.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Add new analysis functions
def get_conversation_summary(conversations):
    all_text = " ".join([conv[0] for conv in conversations])
    words = all_text.lower().split()
    word_freq = Counter(words)
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    word_freq = {k: v for k, v in word_freq.items() if k not in common_words and len(k) > 2}
    return word_freq

def get_sentiment_metrics(conversations):
    sentiments = []
    for conv_text, _ in conversations:
        sentiment, score = analyze_sentiment(conv_text)
        sentiments.append({
            'sentiment': sentiment,
            'score': score,
            'text': conv_text,
            'timestamp': _
        })
    return sentiments

def get_product_recommendations_summary(conversations):
    all_recommendations = []
    for conv_text, _ in conversations:
        current_crm_data = generate_crm_data([conv_text], phone_dataset)
        recommendations = recommend_products(current_crm_data, conv_text)
        all_recommendations.extend(recommendations)
    return Counter(all_recommendations)

# Add new function to get issues summary
def get_issues_summary(conversations):
    """Get summary of troubleshooting issues from conversations"""
    issues = []
    for conv_text, _ in conversations:
        if is_troubleshooting_query(conv_text):  # Now this function will be recognized
            issues.append(conv_text)
    issue_freq = Counter(issues)
    return dict(sorted(issue_freq.items(), key=lambda x: x[1], reverse=True)[:5])

# Add new function for summarization using Gemini
def generate_conversation_summary(conversations, api_key):
    """Generate a summary of all conversations using Gemini"""
    if not conversations:
        return "No conversations to summarize."
        
    all_text = "\n".join([f"User: {conv[0]}" for conv in conversations])
    
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze these customer service conversations and create a concise summary:
        {all_text}

        Create a summary that includes:
        1. Main topics discussed
        2. Common issues or requests
        3. Overall customer sentiment
        4. Key product interests
        5. Notable trends

        Format as a clear, professional paragraph.
        Keep it concise but informative.
        """
        
        response = model.generate_content(prompt)
        return response.text if response.text else "Unable to generate summary."
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Add new function to generate search query summary
def generate_search_query_summary(conversations, api_key):
    """Generate a summary of search queries using Gemini"""
    search_patterns = get_search_summary(conversations)
    if search_patterns['total_searches'] == 0:
        return "No search queries to analyze."
    
    search_text = "\n".join([
        "Product Searches:\n" + "\n".join(search_patterns['recent_searches']['products']),
        "Feature Searches:\n" + "\n".join(search_patterns['recent_searches']['features']),
        "Issue Searches:\n" + "\n".join(search_patterns['recent_searches']['issues'])
    ])
    
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze these search queries and create a concise summary:
        {search_text}

        Create an analysis that includes:
        1. Most common search patterns
        2. Popular product categories
        3. Frequently requested features
        4. Common issues or problems
        5. User preferences and trends

        Format as a clear, professional paragraph.
        Focus on actionable insights about user interests and needs.
        Keep it concise but informative.
        """
        
        response = model.generate_content(prompt)
        return response.text if response.text else "Unable to generate search summary."
    except Exception as e:
        return f"Error generating search summary: {e}"

# Update the calculate_sentiment_shifts function
def calculate_sentiment_shifts(sentiments):
    """Calculate sentiment shifts between consecutive conversations"""
    shifts = []
    for i in range(len(sentiments)-1):
        current = sentiments[i]
        next_sentiment = sentiments[i+1]
        shift = next_sentiment['score'] - current['score']
        
        # Categorize shift magnitude for better visualization
        if abs(shift) < 0.2:
            magnitude = "Minimal"
        elif abs(shift) < 0.5:
            magnitude = "Moderate"
        else:
            magnitude = "Significant"  # Removed incorrect st.warning statements
            
        direction = "🔼" if shift > 0 else "🔽" if shift < 0 else "➡️"
        
        shifts.append({
            'from_time': current['timestamp'],
            'to_time': next_sentiment['timestamp'],
            'shift': shift,
            'magnitude': magnitude,
            'direction': direction,
            'from_sentiment': current['sentiment'],
            'to_sentiment': next_sentiment['sentiment'],
            'from_text': current['text'][:50] + '...',
            'shift_description': f"{direction} {magnitude} ({shift:.2f})"
        })
    return shifts

# Add new function to track search queries
def get_search_summary(conversations):
    """Analyze search patterns from conversations"""
    search_patterns = {
        'products': [],
        'issues': [],
        'features': []
    }
    
    feature_keywords = ['camera', 'battery', 'display', 'processor', 'ram', 'storage']
    
    for conv_text, _ in conversations:
        if any(term in conv_text.lower() for term in ['search', 'find', 'looking for', 'suggest']):
            if is_troubleshooting_query(conv_text):
                search_patterns['issues'].append(conv_text)
            elif any(feature in conv_text.lower() for feature in feature_keywords):
                search_patterns['features'].append(conv_text)
            else:
                search_patterns['products'].append(conv_text)
    
    return {
        'total_searches': len(search_patterns['products']) + len(search_patterns['issues']) + len(search_patterns['features']),
        'product_searches': len(search_patterns['products']),
        'issue_searches': len(search_patterns['issues']),
        'feature_searches': len(search_patterns['features']),
        'recent_searches': search_patterns
    }

# Main Streamlit app
# Load the phone dataset globally
phone_dataset = load_phone_dataset("phone_comparison.csv")

# Create queues for thread communication
audio_queue = queue.Queue()
analysis_queue = queue.Queue()
search_queue = queue.Queue()

# Add new functions for threaded processing
def process_audio(recognizer, audio, transcript_container, session_id):
    """Process audio in a separate thread with longer timeout"""
    try:
        text = recognizer.recognize_google(audio)
        if text:
            save_conversation(session_id, text)
            audio_queue.put((text, transcript_container))
    except Exception as e:
        print(f"Audio processing error: {e}")

def listen_continuously(recognizer, source, stop_event):
    """Listen continuously for speech"""
    try:
        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                return audio
            except sr.WaitTimeoutError:
                continue
    except Exception as e:
        print(f"Continuous listening error: {e}")
        return None

def process_analysis(text, analysis_container):
    """Process analysis in a separate thread"""
    try:
        details = get_conversation_details(text)
        analysis_queue.put((details, analysis_container))
    except Exception as e:
        print(f"Analysis error: {e}")

def process_search(query, is_issue, results_container):
    """Process search in a separate thread"""
    try:
        results = process_object_query(query, phone_dataset)
        search_queue.put((results, is_issue, results_container))
    except Exception as e:
        print(f"Search error: {e}")

def process_voice_input(text, session_id, feedback_placeholder, results_container, timestamp=None):
    """Process voice input and display results immediately"""
    try:
        # Save conversation
        save_conversation(session_id, text)
        
        # Generate CRM data and initialize vector_db
        current_crm_data = generate_crm_data([text], phone_dataset)
        initialize_vector_db(current_crm_data)
        
        # Get all analysis details
        details = get_conversation_details(text)
        is_issue = is_troubleshooting_query(text)
        results = process_object_query(text, phone_dataset)
        
        # Use timestamp for unique keys
        current_time = timestamp or datetime.now().timestamp()
        
        # Display results in organized sections
        with results_container:
            st.markdown("---")
            
            # Show recognized text and sentiment
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"🎤 Recognized: {text}")
            with col2:
                sentiment_score = f"{details['sentiment']} ({details['score']:.2f})"
                st.info(f"😊 Sentiment: {sentiment_score}")
            
            # Show results based on query type
            if is_issue:
                st.subheader("🔧 Troubleshooting Results")
                for i, result in enumerate(results):
                    with st.expander(f"Issue: {result.get('name', 'Problem')}", expanded=True):
                        st.info(f"Type: {result.get('type', 'N/A')}")
                        st.markdown("### Diagnosis")
                        st.write(result.get('diagnosis', 'N/A'))
                        st.markdown("### Solution")
                        st.write(result.get('solution', 'N/A'))
                        if 'parts' in result:
                            st.markdown("### Required Parts")
                            st.write(result.get('parts', 'N/A'))
                        if 'cost' in result:
                            st.info(f"Estimated Cost: {result.get('cost', 'N/A')}")
            else:
                st.subheader("📱 Recommended Products")
                for i, result in enumerate(results):
                    with st.expander(f"📱 {result.get('name', 'Result')}", expanded=True):
                        st.write(f"💰 {result.get('price_category', 'N/A')}")
                        st.write(f"📱 {result.get('display', 'N/A')}")
                        st.write(f"⚙️ {result.get('specs', 'N/A')}")
            
            # Show recommendations
            st.markdown("### 💡 Related Recommendations")
            for rec in details['recommendations']:
                st.write(f"• {rec}")
            
            # Show dynamic questions with unique keys
            st.markdown("### ❓ Follow-up Questions")
            question_cols = st.columns(3)
            for i, question in enumerate(details['questions']):
                with question_cols[i]:
                    if st.button(f"🔍 {question}", 
                               key=f"q_{i}_{current_time}",  # Add timestamp to make key unique
                               use_container_width=True):
                        st.session_state.current_query = question
                        st.rerun()
            
    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")

def main():
    # Initialize session state variables
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'dashboard_pdf_ready' not in st.session_state:
        st.session_state.dashboard_pdf_ready = False
    
    # Initialize executor only if it doesn't exist or was shutdown
    if ('executor' not in st.session_state or 
        st.session_state.executor._shutdown):
        st.session_state.executor = ThreadPoolExecutor(max_workers=3)

    st.title("Speech Recognition, Sentiment Analysis, and Object Handling")

    # Initialize session
    session_id = st.session_state.get("session_id", str(datetime.now().timestamp()))
    st.session_state["session_id"] = session_id

    # Sidebar options including new Dashboard
    option = st.sidebar.radio(
        "Choose an action", 
        ["Dashboard", "Live Recording"]  # Removed separate "Search Query" option
    )

    if option == "Dashboard":
        st.header("Conversation Analysis Dashboard")
        
        # Create tabs for Summary and Detailed View
        summary_tab, detailed_tab = st.tabs(["Conversation Summary", "Detailed Call Reports"])
        
        conversations = fetch_past_conversations(session_id)
        
        if not conversations:
            st.info("No conversation history available.")
        else:
            with summary_tab:
                # Add conversation summary at the top
                st.subheader("Conversation Overview")
                summary = generate_conversation_summary(conversations, api_key)
                st.write(summary)
                
                st.markdown("---")  # Add divider
                
                # Top section with metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Conversation Overview")
                    total_conversations = len(conversations)
                    st.metric("Total Interactions", total_conversations)
                
                with col2:
                    st.subheader("Sentiment Analysis")
                    sentiments = get_sentiment_metrics(conversations)
                    avg_sentiment_score = sum(s['score'] for s in sentiments) / len(sentiments)
                    st.metric("Average Sentiment Score", f"{avg_sentiment_score:.2f}")
                    # Calculate average sentiment
                    sentiment_counts = Counter([s['sentiment'] for s in sentiments])
                    most_common_sentiment = sentiment_counts.most_common(1)[0][0]
                    st.metric("Average Sentiment", most_common_sentiment)
                
                with col3:
                    st.subheader("Top Products")
                    product_freq = get_product_recommendations_summary(conversations)
                    top_products = dict(sorted(product_freq.items(), key=lambda x: x[1], reverse=True)[:3])
                    for product, count in top_products.items():
                        st.write(f"• {product} ({count})")
                
                # Charts section
                st.subheader("Analysis Trends")
                
                # Sentiment Timeline and Heatmap
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_df = pd.DataFrame(sentiments)
                    fig = px.line(
                        sentiment_df,
                        x='timestamp',
                        y='score',
                        title="Sentiment Score Timeline",
                        labels={'score': 'Sentiment Score', 'timestamp': 'Time'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced Sentiment Shift Heatmap
                    shifts = calculate_sentiment_shifts(sentiments)
                    if shifts:
                        st.subheader("Sentiment Shift Analysis")
                        
                        # Create enhanced heatmap
                        fig = go.Figure()
                        
                        # Add heatmap
                        fig.add_trace(go.Heatmap(
                            z=[[s['shift']] for s in shifts],
                            x=[s['from_time'] for s in shifts],
                            y=['Sentiment Change'],
                            colorscale=[
                                [0, '#ff4444'],      # Strong negative (red)
                                [0.25, '#ffab91'],    # Mild negative (light red)
                                [0.5, '#ffffff'],     # Neutral (white)
                                [0.75, '#a5d6a7'],   # Mild positive (light green)
                                [1, '#00c853']       # Strong positive (green)
                            ],
                            showscale=True,
                            colorbar=dict(
                                title='Sentiment Change',
                                titleside='right',
                                thickness=15,
                                len=0.75,
                                ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                                tickvals=[-1, -0.5, 0, 0.5, 1],
                                tickmode='array'
                            )
                        ))
                        
                        # Add markers for shift direction
                        fig.add_trace(go.Scatter(
                            x=[s['from_time'] for s in shifts],
                            y=['Sentiment Change'] * len(shifts),
                            mode='text',
                            text=[s['direction'] for s in shifts],
                            textposition='middle center',
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title={
                                'text': "Conversation Sentiment Shifts",
                                'y': 0.95,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            height=180,
                            margin=dict(l=60, r=30, t=40, b=20),
                            xaxis_title="Time",
                            yaxis=dict(showgrid=False),
                            plot_bgcolor='rgba(255,255,255,0.9)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add detailed shift table with improved formatting
                        st.subheader("Detailed Sentiment Changes")
                        shift_details = pd.DataFrame([{
                            'Time': s['from_time'],
                            'Change': s['shift_description'],
                            'From': f"{s['from_sentiment']} ({s['from_text']})",
                            'To': s['to_sentiment']
                        } for s in shifts])
                        
                        st.dataframe(
                            shift_details,
                            column_config={
                                "Time": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
                                "Change": st.column_config.Column("Shift", width="medium"),
                                "From": st.column_config.Column("Initial Context", width="large"),
                                "To": st.column_config.Column("Final Sentiment", width="medium")
                            },
                            hide_index=True
                        )
                
                with col2:
                    # Replace product recommendations with issues summary
                    st.subheader("Common Issues")
                    issues_freq = get_issues_summary(conversations)
                    if issues_freq:
                        fig = go.Figure(go.Bar(
                            x=list(issues_freq.values()),
                            y=list(issues_freq.keys()),
                            orientation='h'
                        ))
                        fig.update_layout(title="Top Reported Issues")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No issues reported yet")
            
            with summary_tab:
                # Add search analytics section after the metrics
                st.markdown("---")
                st.subheader("Search Analytics")
                
                search_metrics = get_search_summary(conversations)
                
                # Display search metrics in columns
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Searches", search_metrics['total_searches'])
                with c2:
                    st.metric("Product Searches", search_metrics['product_searches'])
                with c3:
                    st.metric("Feature Searches", search_metrics['feature_searches'])
                with c4:
                    st.metric("Issue Searches", search_metrics['issue_searches'])
                
                # Create search patterns visualization
                if search_metrics['total_searches'] > 0:
                    # Create pie chart of search distribution
                    search_dist = {
                        'Products': search_metrics['product_searches'],
                        'Features': search_metrics['feature_searches'],
                        'Issues': search_metrics['issue_searches']
                    }
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(search_dist.keys()),
                        values=list(search_dist.values()),
                        hole=.3,
                        marker_colors=['#2ecc71', '#3498db', '#e74c3c']
                    )])
                    
                    fig.update_layout(
                        title="Search Query Distribution",
                        showlegend=True,
                        height=300,
                        margin=dict(t=30, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show recent searches
                    with st.expander("Recent Search Details"):
                        if search_metrics['recent_searches']['products']:
                            st.subheader("Product Searches")
                            for search in search_metrics['recent_searches']['products'][-3:]:
                                st.info(search)
                        
                        if search_metrics['recent_searches']['features']:
                            st.subheader("Feature-based Searches")
                            for search in search_metrics['recent_searches']['features'][-3:]:
                                st.success(search)
                        
                        if search_metrics['recent_searches']['issues']:
                            st.subheader("Issue-related Searches")
                            for search in search_metrics['recent_searches']['issues'][-3:]:
                                st.warning(search)
                else:
                    st.info("No search queries recorded yet.")
                
                st.markdown("---")
                
                # Continue with existing dashboard content
                # ...existing code...

                # Add new search summary section
                st.markdown("---")
                st.subheader("Search Query Analysis")
                
                # Add two columns for search metrics and summary
                search_col1, search_col2 = st.columns([2, 3])
                
                with search_col1:
                    # Search metrics in a more compact format
                    search_metrics = get_search_summary(conversations)
                    st.metric("Total Searches", search_metrics['total_searches'])
                    search_dist = {
                        'Products': search_metrics['product_searches'],
                        'Features': search_metrics['feature_searches'],
                        'Issues': search_metrics['issue_searches']
                    }
                    
                    # Create donut chart for search distribution
                    if search_metrics['total_searches'] > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(search_dist.keys()),
                            values=list(search_dist.values()),
                            hole=.4,
                            marker_colors=['#2ecc71', '#3498db', '#e74c3c']
                        )])
                        fig.update_layout(
                            title="Search Distribution",
                            showlegend=True,
                            height=250,
                            margin=dict(t=30, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with search_col2:
                    # Generate and display search pattern analysis
                    search_summary = generate_search_query_summary(conversations, api_key)
                    st.markdown("### Search Pattern Analysis")
                    st.write(search_summary)

            with detailed_tab:
                # Existing detailed conversation view
                st.subheader("Detailed Conversation History")
                all_conversation_details = []
                
                for conv_text, timestamp in conversations:
                    # Create an expander for each conversation
                    with st.expander(f"Conversation from {timestamp}"):
                        st.text("Transcript:")
                        st.write(conv_text)
                        
                        # Get or compute conversation details
                        details = get_conversation_details(conv_text)
                        
                        # Store details for PDF
                        all_conversation_details.append({
                            "text": conv_text,
                            "sentiment": details['sentiment'],
                            "score": details['score'],
                            "shift": "N/A",  # Not tracking shifts in dashboard
                            "recommendations": details['recommendations'],
                            "questions": details['questions']
                        })
                        
                        # Display details in organized sections
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Sentiment Analysis")
                            st.write(f"Sentiment: {details['sentiment']}")
                            st.write(f"Score: {round(details['score'], 2)}")
                        
                        with col2:
                            st.subheader("Recommendations")
                            for rec in details['recommendations']:
                                st.write(f"• {rec}")
                        
                        st.subheader("Generated Questions")
                        for i, question in enumerate(details['questions'], 1):
                            st.write(f"{i}. {question}")

                # Add PDF download button only in dashboard with unique key
                if all_conversation_details:
                    pdf_path = generate_pdf(all_conversation_details)
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Complete Conversation History (PDF)",
                            data=pdf_file,
                            file_name="conversation_history.pdf",
                            key="dashboard_pdf_download"
                        )

    elif option == "Live Recording":
        st.header("Interactive Assistant")

        # Add stop event to session state
        if 'stop_listening' not in st.session_state:
            st.session_state.stop_listening = threading.Event()
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""

        # Create main container for results that will be accessible throughout
        results_container = st.container()

        # Create two columns: main interaction and results
        main_col, side_col = st.columns([2, 1])

        with main_col:
            # Create voice-first search interface
            search_container = st.container()
            with search_container:
                # Add text search at the top
                text_query = st.text_input("Search or type your query", 
                    key="text_search",
                    placeholder="Type your question or search query here..."
                )
                
                if text_query:
                    process_voice_input(
                        text_query, 
                        session_id, 
                        None, 
                        results_container,
                        timestamp=datetime.now().timestamp()
                    )
                
                st.markdown("### Or use voice input:")
                
                # Create centered microphone button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.session_state.is_listening:
                        if st.button("⏹️ Stop", help="Stop listening", type="primary", use_container_width=True):
                            st.session_state.stop_listening.set()
                            st.session_state.is_listening = False
                            st.rerun()
                    else:
                        if st.button("🎤 Start Voice Search", help="Start listening", type="primary", use_container_width=True):
                            st.session_state.stop_listening.clear()
                            st.session_state.is_listening = True
                            # Initialize vector_db before starting
                            initialize_vector_db([])
                            st.rerun()
                
                # Voice recording status and container
                voice_container = st.container()
                
                # Continuous voice input handling
                if st.session_state.is_listening:
                    with voice_container:
                        status_placeholder = st.empty()
                        feedback_placeholder = st.empty()
                        status_placeholder.info("🎤 Listening... (Speak your query)")
                        
                        try:
                            recognizer = sr.Recognizer()
                            with sr.Microphone() as source:
                                # Longer ambient noise adjustment
                                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                                
                                while st.session_state.is_listening and not st.session_state.stop_listening.is_set():
                                    try:
                                        feedback_placeholder.info("🎤 Listening actively...")
                                        # Increased timeout and phrase limit for longer sentences
                                        audio = recognizer.listen(source, timeout=20, phrase_time_limit=30)
                                        try:
                                            text = recognizer.recognize_google(audio)
                                            if text:
                                                feedback_placeholder.success(f"Recognized: {text}")
                                                st.session_state.current_query = text
                                                # Process voice input immediately
                                                process_voice_input(text, session_id, feedback_placeholder, results_container)
                                                time.sleep(2)  # Give more time to read results
                                        except sr.UnknownValueError:
                                            feedback_placeholder.warning("Could not understand, please try again")
                                            time.sleep(1)
                                            continue
                                    except sr.WaitTimeoutError:
                                        continue
                                    
                        except Exception as e:
                            st.error(f"Voice input error: {str(e)}")
                            st.session_state.is_listening = False

                # Rest of the code...
                # ...existing code...

# Add new helper function to process queries
def process_query(query, session_id, container):
    """Process search query and display results"""
    # Save the query
    save_conversation(session_id, query)
    
    # Get results and details
    is_issue = is_troubleshooting_query(query)
    results = process_object_query(query, phone_dataset)
    details = get_conversation_details(query)
    
    # Display results
    st.markdown("---")
    
    # Show metadata
    meta_col1, meta_col2 = st.columns([2, 1])
    with meta_col1:
        st.caption(f"Showing results for: {query}")
    with meta_col2:
        sentiment = f"{details['sentiment']} ({details['score']:.2f})"
        st.caption(f"Query sentiment: {sentiment}")
    
    # Rest of your existing results display code...
    # ...existing code...

    # Move executor cleanup to a separate function
    def cleanup():
        if 'executor' in st.session_state:
            try:
                st.session_state.executor.shutdown(wait=True)
            except Exception as e:
                print(f"Error shutting down executor: {e}")
    
    # Register the cleanup function to run on session end
    st.session_state['_cleanup_scheduled'] = cleanup

if __name__ == "__main__":
    initialize_database()
    main()



