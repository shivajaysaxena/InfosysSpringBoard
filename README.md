# 🎙️ Real-Time AI Sales Call Assistant

Welcome to the **Real-Time AI Sales Call Assistant** project! This application provides intelligent real-time analysis of sales conversations, mobile phone recommendations, and customer interaction insights powered by Google's Gemini AI. Built with Streamlit, this tool offers a seamless and interactive experience for sales professionals.

## 🚀 Features

- **🎤 Live Voice Recognition:** Real-time speech-to-text conversion with instant feedback
- **🧠 Sentiment Analysis:** Dynamic emotion tracking and conversation mood analysis
- **💡 Smart Recommendations:** Context-aware product and spare parts suggestions
- **📊 Analytics Dashboard:** Comprehensive visualization of conversation metrics
- **📝 Auto Documentation:** Automatic conversation logging and report generation

## 🛠️ Installation

Follow these steps to set up the project:

1. **🔀 Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **📥 System Requirements:**

   > **⚠️ Python Compatibility**: Requires Python 3.9+ (3.10 recommended)

   **Version Support:**
   - ✅ Python 3.9-3.11: Fully compatible
   - ✅ Python 3.10: Recommended version
   - ❌ Python 3.12+: Not supported by dependencies

   **Check Python Version:**
   ```bash
   python --version  # Windows
   python3 --version  # macOS/Linux
   ```

   **Create Virtual Environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **📦 Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **🔑 Configure Google Gemini API:**
   
   a. Get API key from Google AI Studio
   b. Set environment variable:
   
   **Windows:**
   ```bash
   setx API_KEY "your_gemini_api_key"
   ```
   
   **macOS/Linux:**
   ```bash
   echo "export API_KEY='your_gemini_api_key'" >> ~/.zshrc
   # OR
   echo "export API_KEY='your_gemini_api_key'" >> ~/.bashrc
   ```

## 🖥️ Usage

Launch the application:
```bash
streamlit run app.py
```

## 🎯 Key Features:
1. **🎤 Live Voice Assistant**
Click "Start Voice Search"
Speak your query clearly
View real-time analysis
Use quick filters for common queries

2. **📊 Analytics Dashboard**
Track conversation metrics
Monitor sentiment trends
Analyze product recommendations
Generate PDF reports

## 📁 Project Structure
```
ai-sales-assistant/
├── app.py                # Main application
├── assistant.py          # Core AI functionality
├── requirements.txt      # Dependencies
├── data/
│   ├── phone_comparison.csv    # Product database
│   └── spare_parts.csv         # Spare parts data
└── README.md            # Documentation
```

## 🧰 Core Dependencies

- Streamlit: Interactive UI
- Google Gemini AI: Natural language processing
- SpeechRecognition: Voice input processing
- Pandas: Data management
- FAISS: Vector search
- Plotly: Data visualization
- FPDF: Report generation

## 🤝 Contributing
1. **Fork the repository**
2. **Create feature branch:**
```bash
git checkout -b feature/YourFeature
```
3. **Commit changes:**
```bash
git commit -m "Add: feature description"
```
4. **Push to branch:**
```bash
git push origin feature/YourFeature
```
5. **Open Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📧 Contact
For inquiries or feedback, please contact Shivajay Saxena

<div align="center"> <img src="https://img.icons8.com/color/48/000000/microphone.png" alt="Voice Icon" /> <img src="https://img.icons8.com/color/48/000000/analytics.png" alt="Analytics Icon" /> <img src="https://img.icons8.com/color/48/000000/mobile-payment.png" alt="Sales Icon" /> </div>