# AI-Powered Mobile Phone Assistant

A Streamlit-based conversational assistant that provides intelligent assistance for mobile phone queries, troubleshooting, and recommendations powered by Google's Gemini AI.

## Features

### 1. Live Voice Interaction
- Real-time speech-to-text conversion
- Sentiment analysis of conversations
- Dynamic question generation
- Context-aware product recommendations

### 2. Intelligent Search & Support
- Natural language product search
- Smart troubleshooting assistance
- Spare parts recommendations
- Brand and category-based filtering

### 3. Analytics Dashboard
- Real-time sentiment tracking
- Issue frequency analysis
- Product recommendation trends
- Interactive conversation history
- Detailed call reports
- PDF report generation

### 4. Smart Recommendations
- Dataset-validated product suggestions
- Brand-aware recommendations
- Spare parts matching for repairs
- Context-based filtering

## Technology Stack

- Python 3.9+
- Streamlit
- Google Gemini AI
- Pandas for data management
- SQLite for conversation storage
- SentenceTransformers for text analysis
- Plotly for data visualization
- FPDF for report generation

## Dataset Details

### Phone Dataset (phone_comparison.csv)
- Comprehensive phone specifications
- 45+ latest phone models
- Detailed feature information
- Categorization keywords

### Spare Parts Dataset (spare_parts.csv)
- Common replacement parts
- Compatibility information
- Pricing and availability
- Issue-specific keywords

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your API key:
```bash
export API_KEY=your_google_gemini_api_key
```
4. Run the application:
```bash
streamlit run app.py
```

# Usage Guide

## Live Recording
1. Select "Live Recording" from the sidebar.
2. Click "Start Recording" to begin.
3. Speak your query clearly.
4. View real-time analysis and recommendations.
5. Click "Stop Recording" when finished.

## Search Query
1. Select "Search Query" from the sidebar.
2. Type your query in the search box.
3. View detailed product or spare part information.
4. Explore recommendations and specifications.

## Dashboard
- View conversation analytics.
- Track sentiment trends.
- Monitor common issues.
- Download conversation reports.

## Development

### Adding New Features
1. Update datasets in CSV files.
2. Extend `assistant.py` with new functions.
3. Add UI components in `app.py`.
4. Update documentation.

### Debugging
1. Check console for detailed logs.
2. Verify dataset loading messages.
3. Monitor API responses.
4. Review sentiment analysis results.

