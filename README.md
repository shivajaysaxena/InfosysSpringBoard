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

