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
# from flask import Flask, render_template, request, jsonify, send_file, make_response
# from flask_socketio import SocketIO, emit
# import speech_recognition as sr
# from threading import Thread
# from datetime import datetime
# from assistant import (
#     recognize_speech,
#     analyze_sentiment,
#     generate_embeddings,
#     recommend_products,
#     generate_dynamic_questions,
#     generate_crm_data,
#     initialize_vector_db
# )
# from fpdf import FPDF

# # Initialize Flask and SocketIO
# app = Flask(__name__)
# socketio = SocketIO(app)

# # Globals
# recognizer = sr.Recognizer()
# microphone = sr.Microphone()
# is_listening = False
# transcripts = []  # Store transcripts, sentiments, and recommendations for PDF
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

# @app.route('/')
# def index():
#     session_id = request.cookies.get('session_id', str(datetime.now().timestamp()))
#     response = make_response(render_template('index.html'))
#     response.set_cookie('session_id', session_id)

#     # Initialize CRM data dynamically
#     past_conversations = fetch_past_conversations(session_id)
#     if not past_conversations:
#         print("No past conversations found. Using default CRM data.")
#         past_conversations = ["Looking for a smartphone", "Interested in laptops"]

#     initialize_vector_db(generate_crm_data(past_conversations))
#     return response

# # Update in app.py
# def recognize_speech_in_background(session_id):
#     global is_listening, transcripts, recognizer, microphone
#     previous_sentiment = None  # Track the last sentiment for shift detection

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)
#         while is_listening:
#             try:
#                 audio = recognizer.listen(source)
#                 text = recognizer.recognize_google(audio)
#                 print(f"Transcripted Text: {text}")  # Display transcript in console
#                 socketio.emit('transcript', {'text': text})

#                 # Save conversation
#                 save_conversation(session_id, text)

#                 # Fetch past conversations
#                 past_conversations = fetch_past_conversations(session_id)

#                 # Dynamic CRM update
#                 crm_data = generate_crm_data(past_conversations)

#                 # Sentiment analysis
#                 sentiment, score = analyze_sentiment(text)
#                 sentiment_shift = None
#                 if previous_sentiment and sentiment != previous_sentiment:
#                     sentiment_shift = f"Shifted from {previous_sentiment} to {sentiment}"
#                 previous_sentiment = sentiment
#                 print(f"Sentiment: {sentiment}, Score: {score}, Shift: {sentiment_shift}")  # Debug

#                 # Product Recommendation
#                 recommendations = recommend_products(crm_data, text)

#                 # Dynamic Question Handling
#                 dynamic_questions = generate_dynamic_questions(text)

#                 socketio.emit('analysis', {
#                     'sentiment': sentiment,
#                     'score': score,
#                     'shift': sentiment_shift,
#                     'recommendations': recommendations,
#                     'questions': dynamic_questions
#                 })

#                 # Store for PDF
#                 transcripts.append({
#                     "text": text,
#                     "sentiment": sentiment,
#                     "score": score,
#                     "shift": sentiment_shift,
#                     "recommendations": recommendations,
#                     "questions": dynamic_questions
#                 })

#             except sr.UnknownValueError:
#                 socketio.emit('transcript', {'text': "Could not understand the speech."})
#             except sr.RequestError:
#                 socketio.emit('transcript', {'text': "Speech recognition service unavailable."})

# @socketio.on('start_recording')
# def handle_start_recording():
#     global is_listening
#     if not is_listening:
#         is_listening = True
#         session_id = request.cookies.get('session_id')  # Fetch session ID from the request
#         thread = Thread(target=recognize_speech_in_background, args=(session_id,))
#         thread.daemon = True
#         thread.start()


# @socketio.on('stop_recording')
# def handle_stop_recording():
#     global is_listening
#     is_listening = False

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
#     recommendations = recommend_products(crm_data=[], text=text)  # CRM data dynamically generated
#     dynamic_questions = generate_dynamic_questions(text)

#     return jsonify({
#         'transcript': text,
#         'sentiment': sentiment,
#         'score': score,
#         'recommendations': recommendations,
#         'questions': dynamic_questions
#     })

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
#             f"Recommendations: {', '.join(entry['recommendations'])}\n"
#             f"Questions: {', '.join(entry['questions'])}"
#         )

#     pdf_output = "transcripts_report.pdf"
#     pdf.output(pdf_output)

#     transcripts = []
#     return send_file(pdf_output, as_attachment=True)

# if __name__ == "__main__":
#     initialize_database()
#     os.makedirs("uploads", exist_ok=True)
#     socketio.run(app, debug=True, allow_unsafe_werkzeug=True)





import os
import sqlite3
from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_socketio import SocketIO, emit
import speech_recognition as sr
from threading import Thread
from datetime import datetime
from assistant import (
    recognize_speech,
    analyze_sentiment,
    generate_embeddings,
    generate_recommendations_with_llm,
    generate_dynamic_questions_with_llm,
    generate_crm_data,
    initialize_vector_db
)
from fpdf import FPDF

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Globals
recognizer = sr.Recognizer()
microphone = sr.Microphone()
is_listening = False
transcripts = []  # Store transcripts, sentiments, and recommendations for PDF
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

@app.route('/')
def index():
    session_id = request.cookies.get('session_id', str(datetime.now().timestamp()))
    response = make_response(render_template('index.html'))
    response.set_cookie('session_id', session_id)

    # Initialize CRM data dynamically
    past_conversations = fetch_past_conversations(session_id)
    if not past_conversations:
        print("No past conversations found. Using default CRM data.")
        past_conversations = ["Looking for a smartphone", "Interested in laptops"]

    initialize_vector_db(generate_crm_data(past_conversations))
    return response

# Update in app.py
def recognize_speech_in_background(session_id):
    global is_listening, transcripts, recognizer, microphone
    previous_sentiment = None  # Track the last sentiment for shift detection

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        while is_listening:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                print(f"Transcripted Text: {text}")  # Display transcript in console
                socketio.emit('transcript', {'text': text})

                # Save conversation
                save_conversation(session_id, text)

                # Fetch past conversations
                past_conversations = fetch_past_conversations(session_id)

                # Dynamic CRM update
                crm_data = generate_crm_data(past_conversations)

                # Sentiment analysis
                sentiment, score = analyze_sentiment(text)
                sentiment_shift = None
                if previous_sentiment and sentiment != previous_sentiment:
                    sentiment_shift = f"Shifted from {previous_sentiment} to {sentiment}"
                previous_sentiment = sentiment
                print(f"Sentiment: {sentiment}, Score: {score}, Shift: {sentiment_shift}")  # Debug

                # Product Recommendation with LLM
                recommendations = generate_recommendations_with_llm(text, past_conversations)

                # Dynamic Question Handling with LLM
                dynamic_questions = generate_dynamic_questions_with_llm(text)

                socketio.emit('analysis', {
                    'sentiment': sentiment,
                    'score': score,
                    'shift': sentiment_shift,
                    'recommendations': recommendations,
                    'questions': dynamic_questions
                })

                # Store for PDF
                transcripts.append({
                    "text": text,
                    "sentiment": sentiment,
                    "score": score,
                    "shift": sentiment_shift,
                    "recommendations": recommendations,
                    "questions": dynamic_questions
                })

            except sr.UnknownValueError:
                socketio.emit('transcript', {'text': "Could not understand the speech."})
            except sr.RequestError:
                socketio.emit('transcript', {'text': "Speech recognition service unavailable."})

@socketio.on('start_recording')
def handle_start_recording():
    global is_listening
    if not is_listening:
        is_listening = True
        session_id = request.cookies.get('session_id')  # Fetch session ID from the request
        thread = Thread(target=recognize_speech_in_background, args=(session_id,))
        thread.daemon = True
        thread.start()


@socketio.on('stop_recording')
def handle_stop_recording():
    global is_listening
    is_listening = False

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    text = recognize_speech(file_path)
    sentiment, score = analyze_sentiment(text)
    recommendations = generate_recommendations_with_llm(text, context_data=[])  # CRM data dynamically generated
    dynamic_questions = generate_dynamic_questions_with_llm(text)

    return jsonify({
        'transcript': text,
        'sentiment': sentiment,
        'score': score,
        'recommendations': recommendations,
        'questions': dynamic_questions
    })

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    global transcripts

    # Generate PDF
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
            f"Recommendations: {', '.join(entry['recommendations'])}\n"
            f"Questions: {', '.join(entry['questions'])}"
        )

    pdf_output = "transcripts_report.pdf"
    pdf.output(pdf_output)

    transcripts = []
    return send_file(pdf_output, as_attachment=True)

if __name__ == "__main__":
    initialize_database()
    os.makedirs("uploads", exist_ok=True)
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

