# 🛡️ AI-Powered Email & SMS Spam Detector

A professional spam detection application that combines Machine Learning with Google's Gemini AI for comprehensive spam analysis.

## ✨ Features

- **Advanced ML Classification**: Uses trained model for accurate spam detection
- **Gemini AI Analysis**: Detailed insights on spam types and handling advice
- **30+ Sample Messages**: Test with realistic spam and legitimate messages
- **Professional UI**: Modern, gradient-based design with animations
- **Real-time Processing**: Instant classification and analysis
- **Privacy-Focused**: All processing done locally

## 🚀 Setup Instructions

### 1. Install Dependencies

```powershell
pip install -r requrements.txt
```

### 2. Configure Gemini API (Optional but Recommended)

1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Copy `config.example.py` to `config.py`
3. Add your API key to `config.py`:

```python
GEMINI_API_KEY = "your-actual-api-key-here"
```

**Note**: The app works without Gemini API but won't provide detailed AI analysis.

### 3. Run the Application

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Enter a Message**: Type or paste any email/SMS in the text area
2. **Try Samples**: Click "Random Spam" or "Random Safe" to test with examples
3. **Analyze**: Click "🔍 Analyze Message" button
4. **Review Results**:
   - Get instant spam/legitimate classification
   - View AI-powered detailed analysis (if API configured)
   - See spam indicators and handling advice

## 🔒 Security & Privacy

- ✅ Messages are processed locally
- ✅ No data is stored or transmitted (except to Gemini API for analysis)
- ✅ API keys are kept secure in `config.py` (gitignored)
- ✅ No tracking or analytics

## 📁 File Structure

```
├── app.py                  # Main Streamlit application
├── model1.pkl             # Trained spam detection model
├── vectorizer1.pkl        # Text vectorizer
├── config.py              # API configuration (gitignored)
├── config.example.py      # Config template
├── requrements.txt        # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML Model**: Scikit-learn
- **NLP**: NLTK
- **AI Analysis**: Google Gemini AI
- **Language**: Python 3.8+

## ⚠️ Important Notes

- The `config.py` file is gitignored to protect your API key
- Never commit your actual API key to version control
- The model files (`*.pkl`) should be in the same directory as `app.py`
- NLTK data is downloaded automatically on first run

## 🎯 Sample Messages

The app includes:

- **30 spam samples**: Phishing, scams, financial fraud, etc.
- **30 legitimate samples**: Real-world safe messages

## 📊 Model Information

- **Type**: Binary Classification (Spam/Not Spam)
- **Vectorization**: TF-IDF
- **Preprocessing**: Tokenization, stopword removal, stemming

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

## 📄 License

This project is for educational purposes.

---

**Built with ❤️ using Streamlit and Gemini AI**
