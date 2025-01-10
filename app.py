# import os
# from flask import Flask, render_template, request, jsonify, send_file
# from flask_socketio import SocketIO, emit
# import speech_recognition as sr
# from threading import Thread
# from assistant import recognize_speech, analyze_sentiment
# from fpdf import FPDF

# # Initialize Flask and SocketIO
# app = Flask(__name__)
# socketio = SocketIO(app)

# # Globals
# recognizer = sr.Recognizer()
# microphone = sr.Microphone()
# is_listening = False
# transcripts = []  # Store transcripts, sentiments, and sentiment scores for PDF

# # Speech recognition in the background
# def recognize_speech_in_background():
#     global is_listening, transcripts, recognizer, microphone
#     previous_sentiment = None

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)
#         while is_listening:
#             try:
#                 audio = recognizer.listen(source)
#                 text = recognizer.recognize_google(audio)
#                 print(f"Transcripted Text: {text}")  # Display transcript in console
#                 socketio.emit('transcript', {'text': text})

#                 # Sentiment analysis
#                 sentiment, score = analyze_sentiment(text)
#                 sentiment_shift = None
#                 if previous_sentiment and sentiment != previous_sentiment:
#                     sentiment_shift = f"Shifted from {previous_sentiment} to {sentiment}"
#                 previous_sentiment = sentiment

#                 print(f"Sentiment: {sentiment}, Score: {score}")  # Display sentiment and score in console
#                 socketio.emit('sentiment', {'sentiment': sentiment, 'score': score, 'shift': sentiment_shift})

#                 # Store for PDF
#                 transcripts.append({
#                     "text": text,
#                     "sentiment": sentiment,
#                     "score": score,
#                     "shift": sentiment_shift
#                 })
#             except sr.UnknownValueError:
#                 socketio.emit('transcript', {'text': "Could not understand the speech."})
#             except sr.RequestError:
#                 socketio.emit('transcript', {'text': "Speech recognition service unavailable."})

# def start_listening():
#     global is_listening
#     if not is_listening:  # Check if it's already listening
#         is_listening = True
#         thread = Thread(target=recognize_speech_in_background)
#         thread.daemon = True
#         thread.start()

# def stop_listening():
#     global is_listening
#     is_listening = False

# @app.route('/')
# def index():
#     return render_template('index.html')

# @socketio.on('start_recording')
# def handle_start_recording():
#     start_listening()

# @socketio.on('stop_recording')
# def handle_stop_recording():
#     stop_listening()

# @app.route('/download_pdf', methods=['GET'])
# def download_pdf():
#     global transcripts

#     # Generate PDF
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Speech Analysis Report", ln=True, align='C')

#     for entry in transcripts:
#         pdf.ln(10)
#         pdf.multi_cell(0, 10, 
#             f"Transcript: {entry['text']}\n"
#             f"Sentiment: {entry['sentiment']}\n"
#             f"Score: {entry['score']}\n"
#             f"Shift: {entry.get('shift', 'None')}"
#         )

#     pdf_output = "transcripts_report.pdf"
#     pdf.output(pdf_output)

#     # Reset transcripts after generating PDF
#     transcripts = []

#     return send_file(pdf_output, as_attachment=True)

# @app.route('/upload', methods=['POST'])
# def upload_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file uploaded"}), 400

#     file = request.files['audio']
#     file_path = f"uploads/{file.filename}"
#     os.makedirs("uploads", exist_ok=True)
#     file.save(file_path)

#     text = recognize_speech(file_path)
#     sentiment, score = analyze_sentiment(text)

#     return jsonify({'transcript': text, 'sentiment': sentiment, 'score': score})

# if __name__ == "__main__":
#     os.makedirs("uploads", exist_ok=True)
#     socketio.run(app, debug=True, allow_unsafe_werkzeug=True)





# m3
# import os
# import sqlite3
# import streamlit as st
# import speech_recognition as sr
# from datetime import datetime
# from assistant import (
#     recognize_speech,
#     analyze_sentiment,
#     recommend_products,
#     generate_dynamic_questions,
#     generate_crm_data,
#     initialize_vector_db,
# )
# from fpdf import FPDF
# import time

# # Initialize database path
# database_path = "conversations.db"

# # Initialize the database
# def initialize_database():
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute(""" 
#         CREATE TABLE IF NOT EXISTS conversations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id TEXT,
#             text TEXT,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     conn.commit()
#     conn.close()

# # Save conversation to the database
# def save_conversation(session_id, text):
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO conversations (session_id, text) VALUES (?, ?)", (session_id, text))
#     conn.commit()
#     conn.close()

# # Fetch past conversations for the session
# def fetch_past_conversations(session_id):
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute("SELECT text FROM conversations WHERE session_id = ?", (session_id,))
#     rows = cursor.fetchall()
#     conn.close()
#     return [row[0] for row in rows]

# # Generate PDF
# def generate_pdf(transcripts):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Speech Analysis Report", ln=True, align='C')

#     for entry in transcripts:
#         pdf.ln(10)
#         pdf.multi_cell(0, 10,
#             f"Transcript: {entry['text']}\n"
#             f"Sentiment: {entry['sentiment']}\n"
#             f"Score: {entry['score']}\n"
#             f"Shift: {entry['shift']}\n"
#             f"Recommendations: {', '.join(entry['recommendations'])}\n"
#             f"Questions: {', '.join(entry['questions'])} "
#         )

#     pdf_output = "transcripts_report.pdf"
#     pdf.output(pdf_output)
#     return pdf_output

# # Main Streamlit app
# def main():
#     st.title("Speech Recognition and Sentiment Analysis")

#     # Initialize session
#     session_id = st.session_state.get("session_id", str(datetime.now().timestamp()))
#     st.session_state["session_id"] = session_id

#     # Initialize CRM data dynamically
#     past_conversations = fetch_past_conversations(session_id)
#     if not past_conversations:
#         past_conversations = ["Looking for a smartphone", "Interested in laptops"]
#     initialize_vector_db(generate_crm_data(past_conversations))

#     # Sidebar for file upload or recording
#     option = st.sidebar.radio("Choose an action", ["Live Recording", "Upload Audio File"])

#     transcripts = []

#     if option == "Live Recording":
#         st.header("Live Recording")

#         recognizer = sr.Recognizer()
#         microphone = sr.Microphone()

#         st.write("Start speaking...")

#         # Buttons for Start and Stop recording on the same line
#         col1, col2 = st.columns(2)
#         with col1:
#             if not st.session_state.get("recording_active", False):
#                 if st.button("Start Recording"):
#                     st.session_state.recording_active = True
#                     st.session_state["previous_sentiment"] = None  # Reset previous sentiment
#                     st.write("Recording started...")
#         with col2:
#             if st.session_state.get("recording_active", False):
#                 if st.button("Stop Recording"):
#                     st.session_state.recording_active = False
#                     st.write("Recording stopped.")

#         # Start Recording
#         if st.session_state.get("recording_active", False):
#             with microphone as source:
#                 recognizer.adjust_for_ambient_noise(source)
#                 try:
#                     st.info("Listening... Speak now!")

#                     while st.session_state.get("recording_active", False):
#                         try:
#                             # Capture audio in small chunks
#                             audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increased timeout to 10 sec
#                             text = recognizer.recognize_google(audio)

#                             # Debugging statements
#                             print(f"Transcript: {text}")

#                             if text:
#                                 st.session_state["latest_transcript"] = text
#                                 save_conversation(session_id, text)

#                                 sentiment, score = analyze_sentiment(text)

#                                 # Debugging statements
#                                 print(f"Sentiment: {sentiment}")
#                                 print(f"Score: {score}")

#                                 # Sentiment shift logic
#                                 previous_sentiment = st.session_state.get("previous_sentiment", None)
#                                 sentiment_shift = "None"

#                                 if previous_sentiment and previous_sentiment != sentiment:
#                                     sentiment_shift = "Changed" if previous_sentiment != sentiment else "Same"
                                
#                                 # Update previous sentiment for the next comparison
#                                 st.session_state["previous_sentiment"] = sentiment

#                                 crm_data = generate_crm_data(fetch_past_conversations(session_id))
#                                 recommendations = recommend_products(crm_data, text)
#                                 dynamic_questions = generate_dynamic_questions(text, api_key)

#                                 st.session_state["latest_sentiment"] = sentiment
#                                 st.session_state["latest_score"] = score
#                                 st.session_state["latest_recommendations"] = recommendations
#                                 st.session_state["latest_questions"] = dynamic_questions

#                                 # Display analysis
#                                 st.write(f"Transcript: {text}")
#                                 st.write(f"Sentiment: {sentiment}")
#                                 st.write(f"Score: {score}")
#                                 st.write(f"Recommendations: {', '.join(recommendations)}")
#                                 st.write(f"Dynamic Questions: {', '.join(dynamic_questions)}")
#                                 st.write(f"Sentiment Shift: {sentiment_shift}")

#                                 # Append to transcripts list for PDF generation
#                                 transcripts.append({
#                                     "text": text,
#                                     "sentiment": sentiment,
#                                     "score": score,
#                                     "shift": sentiment_shift,
#                                     "recommendations": recommendations,
#                                     "questions": dynamic_questions,
#                                 })

#                             time.sleep(1)  # Sleep to avoid overwhelming the processor

#                         except sr.WaitTimeoutError:
#                             print("No speech detected. Waiting for the next phrase...")
#                             time.sleep(1)  # Wait before trying again

#                 except sr.UnknownValueError:
#                     st.error("Could not understand the speech.")
#                     print("Error: Could not understand the speech.")
#                 except sr.RequestError:
#                     st.error("Speech recognition service unavailable.")
#                     print("Error: Speech recognition service unavailable.")

#         # Download PDF after stopping recording
#         if not st.session_state.get("recording_active", False) and transcripts:
#             pdf_path = generate_pdf(transcripts)
#             with open(pdf_path, "rb") as pdf_file:
#                 st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

#     elif option == "Upload Audio File":
#         st.header("Upload Audio File")
#         uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
#         if uploaded_file:
#             file_path = os.path.join("uploads", uploaded_file.name)
#             os.makedirs("uploads", exist_ok=True)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             text = recognize_speech(file_path)
#             sentiment, score = analyze_sentiment(text)
#             crm_data = generate_crm_data(past_conversations)
#             recommendations = recommend_products(crm_data, text)
#             dynamic_questions = generate_dynamic_questions(text, api_key)

#             st.success(f"Transcript: {text}")
#             st.write(f"Sentiment: {sentiment}")
#             st.write(f"Score: {score}")
#             st.write("Recommendations:")
#             st.write(recommendations)
#             st.write("Dynamic Questions:")
#             st.write(dynamic_questions)

#             transcripts.append({
#                 "text": text,
#                 "sentiment": sentiment,
#                 "score": score,
#                 "shift": "None",
#                 "recommendations": recommendations,
#                 "questions": dynamic_questions,
#             })

#     # Download PDF
#     if transcripts:
#         pdf_path = generate_pdf(transcripts)
#         with open(pdf_path, "rb") as pdf_file:
#             st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

# if __name__ == "__main__":
#     initialize_database()
#     main()













# import os
# import sqlite3
# import streamlit as st
# import speech_recognition as sr
# from datetime import datetime
# from assistant import (
#     recognize_speech,
#     analyze_sentiment,
#     recommend_products,
#     generate_dynamic_questions,
#     generate_crm_data,
#     initialize_vector_db,
# )
# from fpdf import FPDF
# import time

# # Initialize database path
# database_path = "conversations.db"

# # Initialize the database
# def initialize_database():
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute(""" 
#         CREATE TABLE IF NOT EXISTS conversations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             session_id TEXT,
#             text TEXT,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     conn.commit()
#     conn.close()

# # Save conversation to the database
# def save_conversation(session_id, text):
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO conversations (session_id, text) VALUES (?, ?)", (session_id, text))
#     conn.commit()
#     conn.close()

# # Fetch past conversations for the session
# def fetch_past_conversations(session_id):
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#     cursor.execute("SELECT text FROM conversations WHERE session_id = ?", (session_id,))
#     rows = cursor.fetchall()
#     conn.close()
#     return [row[0] for row in rows]

# # Generate PDF
# def generate_pdf(transcripts):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Speech Analysis Report", ln=True, align='C')

#     for entry in transcripts:
#         pdf.ln(10)
#         pdf.multi_cell(0, 10,
#             f"Transcript: {entry['text']}\n"
#             f"Sentiment: {entry['sentiment']}\n"
#             f"Score: {entry['score']}\n"
#             f"Shift: {entry['shift']}\n"
#             f"Recommendations: {', '.join(entry['recommendations'])}\n"
#             f"Questions: {', '.join(entry['questions'])} "
#         )

#     pdf_output = "transcripts_report.pdf"
#     pdf.output(pdf_output)
#     return pdf_output

# # Main Streamlit app
# def main():
#     st.title("Speech Recognition and Sentiment Analysis")

#     # Initialize session
#     session_id = st.session_state.get("session_id", str(datetime.now().timestamp()))
#     st.session_state["session_id"] = session_id

#     # Initialize CRM data dynamically
#     past_conversations = fetch_past_conversations(session_id)
#     if not past_conversations:
#         past_conversations = ["Looking for a smartphone", "Interested in laptops"]
#     initialize_vector_db(generate_crm_data(past_conversations))

#     # Sidebar for file upload or recording
#     option = st.sidebar.radio("Choose an action", ["Live Recording", "Upload Audio File"])

#     transcripts = []

#     if option == "Live Recording":
#         st.header("Live Recording")

#         recognizer = sr.Recognizer()
#         microphone = sr.Microphone()

#         st.write("Start speaking...")

#         # Buttons for Start and Stop recording on the same line
#         col1, col2 = st.columns(2)
#         with col1:
#             if not st.session_state.get("recording_active", False):
#                 if st.button("Start Recording"):
#                     st.session_state.recording_active = True
#                     st.session_state["previous_sentiment"] = None  # Reset previous sentiment
#                     st.write("Recording started...")
#         with col2:
#             if st.session_state.get("recording_active", False):
#                 if st.button("Stop Recording"):
#                     st.session_state.recording_active = False
#                     st.write("Recording stopped.")

#         # Start Recording
#         if st.session_state.get("recording_active", False):
#             with microphone as source:
#                 recognizer.adjust_for_ambient_noise(source)
#                 try:
#                     st.info("Listening... Speak now!")

#                     while st.session_state.get("recording_active", False):
#                         try:
#                             # Capture audio in small chunks
#                             audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)  # Increased timeout to 10 sec
#                             text = recognizer.recognize_google(audio)

#                             # Debugging statements
#                             print(f"Transcript: {text}")

#                             if text:
#                                 st.session_state["latest_transcript"] = text
#                                 save_conversation(session_id, text)

#                                 sentiment, score = analyze_sentiment(text)

#                                 # Debugging statements
#                                 print(f"Sentiment: {sentiment}")
#                                 print(f"Score: {score}")

#                                 # Sentiment shift logic
#                                 previous_sentiment = st.session_state.get("previous_sentiment", None)
#                                 sentiment_shift = "None"

#                                 if previous_sentiment and previous_sentiment != sentiment:
#                                     sentiment_shift = "Changed" if previous_sentiment != sentiment else "Same"
                                
#                                 # Update previous sentiment for the next comparison
#                                 st.session_state["previous_sentiment"] = sentiment

#                                 crm_data = generate_crm_data(fetch_past_conversations(session_id))
#                                 recommendations = recommend_products(crm_data, text)
#                                 dynamic_questions = generate_dynamic_questions(text, api_key)

#                                 st.session_state["latest_sentiment"] = sentiment
#                                 st.session_state["latest_score"] = score
#                                 st.session_state["latest_recommendations"] = recommendations
#                                 st.session_state["latest_questions"] = dynamic_questions

#                                 # Display analysis
#                                 st.write(f"Transcript: {text}")
#                                 st.write(f"Sentiment: {sentiment}")
#                                 st.write(f"Score: {score}")
#                                 st.write(f"Recommendations: {', '.join(recommendations)}")
#                                 st.write(f"Dynamic Questions: {', '.join(dynamic_questions)}")
#                                 st.write(f"Sentiment Shift: {sentiment_shift}")

#                                 # Append to transcripts list for PDF generation
#                                 transcripts.append({
#                                     "text": text,
#                                     "sentiment": sentiment,
#                                     "score": score,
#                                     "shift": sentiment_shift,
#                                     "recommendations": recommendations,
#                                     "questions": dynamic_questions,
#                                 })

#                             time.sleep(1)  # Sleep to avoid overwhelming the processor

#                         except sr.WaitTimeoutError:
#                             print("No speech detected. Waiting for the next phrase...")
#                             time.sleep(1)  # Wait before trying again

#                 except sr.UnknownValueError:
#                     st.error("Could not understand the speech.")
#                     print("Error: Could not understand the speech.")
#                 except sr.RequestError:
#                     st.error("Speech recognition service unavailable.")
#                     print("Error: Speech recognition service unavailable.")

#         # Show download button only after recording stops and transcripts are available
#         if not st.session_state.get("recording_active", False) and transcripts:
#             pdf_path = generate_pdf(transcripts)
#             with open(pdf_path, "rb") as pdf_file:
#                 st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

#     elif option == "Upload Audio File":
#         st.header("Upload Audio File")
#         uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
#         if uploaded_file:
#             file_path = os.path.join("uploads", uploaded_file.name)
#             os.makedirs("uploads", exist_ok=True)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             text = recognize_speech(file_path)
#             sentiment, score = analyze_sentiment(text)
#             crm_data = generate_crm_data(past_conversations)
#             recommendations = recommend_products(crm_data, text)
#             dynamic_questions = generate_dynamic_questions(text, api_key)

#             st.success(f"Transcript: {text}")
#             st.write(f"Sentiment: {sentiment}")
#             st.write(f"Score: {score}")
#             st.write("Recommendations:")
#             st.write(recommendations)
#             st.write("Dynamic Questions:")
#             st.write(dynamic_questions)

#             transcripts.append({
#                 "text": text,
#                 "sentiment": sentiment,
#                 "score": score,
#                 "shift": "None",
#                 "recommendations": recommendations,
#                 "questions": dynamic_questions,
#             })

#     # Make sure PDF is available for download after recording is stopped
#     if transcripts and not st.session_state.get("recording_active", False):
#         pdf_path = generate_pdf(transcripts)
#         with open(pdf_path, "rb") as pdf_file:
#             st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

# if __name__ == "__main__":
#     initialize_database()
#     main()








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
    load_phone_dataset
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
    st.title("Speech Recognition and Sentiment Analysis")

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
    option = st.sidebar.radio("Choose an action", ["Live Recording", "Upload Audio File"])

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

                                crm_data = generate_crm_data(fetch_past_conversations(session_id), phone_dataset)
                                recommendations = recommend_products(crm_data, text)
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

    # Make sure PDF is available for download after recording is stopped
    if transcripts and not st.session_state.get("recording_active", False):
        pdf_path = generate_pdf(transcripts)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(label="Download Analysis Report (PDF)", data=pdf_file, file_name="analysis_report.pdf")

if __name__ == "__main__":
    initialize_database()
    main()

