from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')
import sqlite3
import streamlit as st
import speech_recognition as sr
from datetime import datetime
from assistant import (
    recognize_speech,
    analyze_sentiment,
    recommend_products,
    generate_dynamic_questions,
    generate_crm_data,
    initialize_vector_db,
    load_phone_dataset,
    process_object_query
)
from fpdf import FPDF
import time

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
    cursor.execute("SELECT text FROM conversations WHERE session_id = ?", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

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

# Main Streamlit app
# Load the phone dataset globally
phone_dataset = load_phone_dataset("phone_comparison.csv")

def main():
    st.title("Speech Recognition, Sentiment Analysis, and Object Handling")

    # Initialize session
    session_id = st.session_state.get("session_id", str(datetime.now().timestamp()))
    st.session_state["session_id"] = session_id

    # Fetch past conversations
    past_conversations = fetch_past_conversations(session_id)
    if not past_conversations:
        past_conversations = ["Looking for a smartphone", "Interested in laptops"]

    # Generate CRM data dynamically from the dataset
    crm_data = generate_crm_data(past_conversations, phone_dataset)
    print("Generated CRM Data:", crm_data)
    if crm_data:
        initialize_vector_db(crm_data)
    else:
        print("No CRM data generated. Skipping vector database initialization.")


    # Sidebar for file upload or recording
    option = st.sidebar.radio("Choose an action", ["Live Recording", "Upload Audio File", "Search Query"])

    transcripts = []

    if option == "Live Recording":
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
                                transcripts.append({
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

        # Show download button only after recording stops and transcripts are available
        if not st.session_state.get("recording_active", False) and transcripts:
            # Generate the PDF report
            pdf_path = generate_pdf(transcripts)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

    elif option == "Upload Audio File":
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if uploaded_file:
            file_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            text = recognize_speech(file_path)
            sentiment, score = analyze_sentiment(text)
            crm_data = generate_crm_data(past_conversations, phone_dataset)
            recommendations = recommend_products(crm_data, text)
            dynamic_questions = generate_dynamic_questions(text, api_key)

            st.success(f"Transcript: {text}")
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Score: {score}")
            st.write("Recommendations:")
            st.write(recommendations)
            st.write("Dynamic Questions:")
            st.write(dynamic_questions)

            transcripts.append({
                "text": text,
                "sentiment": sentiment,
                "score": score,
                "shift": "None",
                "recommendations": recommendations,
                "questions": dynamic_questions,
            })
    
    elif option == "Search Query":
        st.sidebar.header("Phone Search")
        search_query = st.sidebar.text_input("Search for phones", "")
        if search_query:
            phone_results = process_object_query(search_query, phone_dataset)
            
            if phone_results:
                st.sidebar.subheader("Search Results")
                for phone in phone_results:
                    with st.sidebar.expander(phone['name']):
                        st.write("📱 Display:", phone['display'])
                        st.write("📸 Camera:", phone['camera'])
                        st.write("⚙️ Specifications:", phone['specs'])
                        st.write("🔄 OS:", phone['os'])
            else:
                st.sidebar.warning("No matching phones found.")

    # Make sure PDF is available for download after recording is stopped
    if transcripts and not st.session_state.get("recording_active", False):
        pdf_path = generate_pdf(transcripts)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

if __name__ == "__main__":
    initialize_database()
    main()

