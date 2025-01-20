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

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')

# Initialize assistant before using any functionality
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

# Update the calculate_sentiment_shifts function for better visualization
def calculate_sentiment_shifts(sentiments):
    """Calculate sentiment shifts between consecutive conversations"""
    shifts = []
    for i in range(len(sentiments)-1):
        current = sentiments[i]
        next_sentiment = sentiments[i+1]
        shift = next_sentiment['score'] - current['score']
        # Add more context to shifts
        shifts.append({
            'from_time': current['timestamp'],
            'to_time': next_sentiment['timestamp'],
            'shift': shift,
            'magnitude': abs(shift),
            'direction': 'Positive' if shift > 0 else 'Negative' if shift < 0 else 'Neutral',
            'from_text': current['text'][:50] + '...',  # Include conversation snippet
            'from_sentiment': current['sentiment'],
            'to_sentiment': next_sentiment['sentiment']
        })
    return shifts

# Main Streamlit app
# Load the phone dataset globally
phone_dataset = load_phone_dataset("phone_comparison.csv")

def main():
    # Initialize session state variables
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = []
    if 'dashboard_pdf_ready' not in st.session_state:
        st.session_state.dashboard_pdf_ready = False

    st.title("Speech Recognition, Sentiment Analysis, and Object Handling")

    # Initialize session
    session_id = st.session_state.get("session_id", str(datetime.now().timestamp()))
    st.session_state["session_id"] = session_id

    # Sidebar options including new Dashboard
    option = st.sidebar.radio(
        "Choose an action", 
        ["Dashboard", "Live Recording", "Search Query"]
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
                        
                        # Create heatmap
                        shift_df = pd.DataFrame(shifts)
                        fig = go.Figure(data=go.Heatmap(
                            z=[[s['shift']] for s in shifts],
                            x=[s['from_time'] for s in shifts],
                            y=['Sentiment Change'],
                            colorscale=[
                                [0, 'red'],     # Strong negative
                                [0.4, 'yellow'], # Mild negative
                                [0.5, 'white'],  # Neutral
                                [0.6, 'lightgreen'], # Mild positive
                                [1, 'darkgreen']    # Strong positive
                            ],
                            text=[[f"{s['shift']:.2f}<br>({s['direction']})"] for s in shifts],
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            colorbar=dict(
                                title='Sentiment Shift',
                                ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                                tickvals=[-1, -0.5, 0, 0.5, 1]
                            )
                        ))
                        
                        fig.update_layout(
                            title="Conversation Sentiment Shifts",
                            xaxis_title="Time",
                            height=200,
                            margin=dict(l=60, r=30, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add shift details table
                        st.subheader("Shift Details")
                        shift_details = pd.DataFrame([{
                            'Time': s['from_time'],
                            'From': s['from_sentiment'],
                            'To': s['to_sentiment'],
                            'Change': f"{s['shift']:.2f}",
                            'Context': s['from_text']
                        } for s in shifts])
                        
                        st.dataframe(
                            shift_details,
                            column_config={
                                "Time": "Timestamp",
                                "From": "Initial Sentiment",
                                "To": "Next Sentiment",
                                "Change": "Sentiment Shift",
                                "Context": "Conversation Snippet"
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
        st.header("Live Recording")

        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        st.write("Start speaking...")

        # Buttons for Start and Stop recording on the same line
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.get("recording_active", False):
                if st.button("Start Recording"):
                    st.session_state.recording_active = True
                    st.session_state["previous_sentiment"] = None  # Reset previous sentiment
                    st.write("Recording started...")
        with col2:
            if st.session_state.get("recording_active", False):
                if st.button("Stop Recording"):
                    st.session_state.recording_active = False
                    st.write("Recording stopped.")

        # Start Recording
        if st.session_state.get("recording_active", False):
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    st.info("Listening... Speak now!")

                    while st.session_state.get("recording_active", False):
                        try:
                            # Capture audio in small chunks
                            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increased timeout to 10 sec
                            text = recognizer.recognize_google(audio)

                            # Debugging statements
                            print(f"Transcript: {text}")

                            if text:
                                st.session_state["latest_transcript"] = text
                                # Save conversation but use only current text for recommendations
                                save_conversation(session_id, text)

                                sentiment, score = analyze_sentiment(text)

                                # Debugging statements
                                print(f"Sentiment: {sentiment}")
                                print(f"Score: {score}")

                                # Sentiment shift logic
                                previous_sentiment = st.session_state.get("previous_sentiment", None)
                                sentiment_shift = "None"

                                if previous_sentiment and previous_sentiment != sentiment:
                                    sentiment_shift = "Changed" if previous_sentiment != sentiment else "Same"
                                
                                # Update previous sentiment for the next comparison
                                st.session_state["previous_sentiment"] = sentiment

                                # Generate CRM data using only the current text
                                current_crm_data = generate_crm_data([text], phone_dataset)
                                initialize_vector_db(current_crm_data)
                                recommendations = recommend_products(current_crm_data, text)
                                dynamic_questions = generate_dynamic_questions(text, api_key)

                                st.session_state["latest_sentiment"] = sentiment
                                st.session_state["latest_score"] = score
                                st.session_state["latest_recommendations"] = recommendations
                                st.session_state["latest_questions"] = dynamic_questions

                                # Display analysis
                                st.write(f"Transcript: {text}")
                                st.write(f"Sentiment: {sentiment}")
                                st.write(f"Score: {score}")
                                st.write(f"Recommendations: {', '.join(recommendations)}")
                                st.write(f"Dynamic Questions: {', '.join(dynamic_questions)}")
                                st.write(f"Sentiment Shift: {sentiment_shift}")

                                # Append to transcripts list for PDF generation
                                st.session_state.transcripts.append({
                                    "text": text,
                                    "sentiment": sentiment,
                                    "score": score,
                                    "shift": sentiment_shift,
                                    "recommendations": recommendations,
                                    "questions": dynamic_questions,
                                })

                            time.sleep(1)  # Sleep to avoid overwhelming the processor

                        except sr.WaitTimeoutError:
                            print("No speech detected. Waiting for the next phrase...")
                            time.sleep(1)  # Wait before trying again

                except sr.UnknownValueError:
                    st.error("Could not understand the speech.")
                    print("Error: Could not understand the speech.")
                except sr.RequestError:
                    st.error("Speech recognition service unavailable.")
                    print("Error: Speech recognition service unavailable.")

    elif option == "Search Query":
        st.sidebar.header("Product & Support Search")
        search_query = st.sidebar.text_input("Search for phones or describe issues", "")
        if search_query:
            results = process_object_query(search_query, phone_dataset)
            
            if results:
                st.sidebar.subheader("Search Results")
                for result in results:
                    if 'specs' in result and 'display' in result:  # Phone result
                        with st.sidebar.expander(f"📱 {result['name']}"):
                            st.write("📌 Display:", result['display'])
                            st.write("⚙️ Specifications:", result['specs'])
                            st.write("💰 Price Range:", result.get('camera', 'N/A'))
                            st.write("ℹ️ Recommendation:", result.get('os', 'N/A'))
                    else:  # Spare part or other result
                        with st.sidebar.expander(f"🔧 {result['name']}"):
                            st.write("🛠️ Type:", result.get('type', 'N/A'))
                            st.write("📱 Details:", result.get('compatible', 'N/A'))
                            st.write("💰 Price:", result.get('price', 'N/A'))
                            st.write("✅ Availability:", result.get('availability', 'N/A'))
                            st.write("ℹ️ Recommendation:", result.get('message', 'N/A'))
            else:
                st.sidebar.warning("No matching results found.")

if __name__ == "__main__":
    initialize_database()
    main()

