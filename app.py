import streamlit as st
import pickle
import string
import nltk
import time
import random
import google.generativeai as genai
import os
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(
    page_title="Email Spam Detection using NLP & Machine Learning",
    page_icon="📧",
    layout="wide",
)

# ------------------- Load API Key -------------------
def load_api_key():
    config_file = Path("config.py")
    if config_file.exists():
        try:
            import config
            return config.GEMINI_API_KEY
        except:
            return None
    return None

GEMINI_API_KEY = load_api_key()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------- Sample Messages -------------------
SPAM_SAMPLES = [
    "URGENT! Your account will be suspended. Click here immediately to verify your identity.",
    "Congratulations! You've won a $5000 gift card. Claim now at this link!",
    "Dear Customer, Your package is waiting. Pay $2.99 shipping fee to receive it.",
    "You have been selected for a special loan offer. Apply now with no credit check!",
    "WINNER ALERT! You've won the lottery. Send your bank details to claim $1,000,000.",
    "Your Amazon order #12345 has been placed. If you didn't order, click here NOW.",
    "Limited time offer! Get rich quick with our investment program. 500% returns guaranteed!",
    "Your Apple ID has been locked. Reset password immediately to avoid permanent suspension.",
    "Hot singles in your area want to meet you! Click here to chat now.",
    "Congratulations! You qualify for a FREE iPhone 15 Pro. Just pay shipping $4.99.",
    "FINAL NOTICE: Your tax refund of $3,500 is pending. Claim within 24 hours.",
    "We tried to deliver your package but you weren't home. Reschedule delivery here.",
    "Your Netflix subscription payment failed. Update payment method to avoid cancellation.",
    "Bank Alert: Suspicious activity detected. Verify your account immediately.",
    "You've been pre-approved for a $50,000 personal loan. No documents required!",
    "Claim your FREE trial of weight loss pills. Lose 30 pounds in 30 days guaranteed!",
    "Microsoft Security Alert: Your computer has been infected. Call this number NOW.",
    "You have (3) pending messages. Click here to read your messages.",
    "URGENT: Your PayPal account has been limited. Confirm your identity now.",
    "Get a FREE cruise vacation! You've been specially selected. Book now!",
    "Your electricity bill is overdue. Pay now to avoid disconnection within 48 hours.",
    "Work from home and earn $5000/week! No experience needed. Start today!",
    "Your social security number has been suspended due to suspicious activity.",
    "Free gift card waiting for you! Complete this survey to claim $100 Amazon card.",
    "Your Bitcoin wallet needs verification. Click here to secure your account.",
    "LAST CHANCE: Buy medications online at 90% discount. No prescription needed!",
    "You've inherited $2.5 million from a distant relative. Contact us for details.",
    "Your WhatsApp verification code is 123456. If you didn't request this, click here.",
    "Congratulations! You've been approved for debt relief. Reduce debt by 80%.",
    "URGENT: Your car warranty is about to expire. Renew now to avoid losing coverage."
]

NOT_SPAM_SAMPLES = [
    "Hi! Are we still meeting for coffee at 3pm today? Let me know if you need to reschedule.",
    "Mom, I'll be home late tonight. Got stuck in a meeting. Don't wait for dinner.",
    "Thanks for the great presentation today! The client was really impressed with your work.",
    "Hey, just wanted to remind you about Sarah's birthday party this Saturday at 7pm.",
    "Your flight BA123 to New York is confirmed for tomorrow at 10:45 AM. Check-in opens now.",
    "Meeting rescheduled to 2pm in Conference Room B. Please bring the quarterly reports.",
    "Your Amazon order has been delivered. Thank you for shopping with us!",
    "Dr. Smith's office calling to confirm your appointment on Friday at 2:30 PM.",
    "Happy birthday! Hope you have an amazing day. Let's celebrate this weekend!",
    "Your electricity bill for March is $87.50. Payment is due by April 15th.",
    "Team lunch at the Italian restaurant tomorrow at 12:30. Hope you can make it!",
    "Your prescription is ready for pickup at CVS Pharmacy on Main Street.",
    "Class reminder: Assignment due this Friday. Office hours tomorrow 2-4pm if you need help.",
    "Thanks for your purchase! Your order #5678 will ship within 2 business days.",
    "Gym membership renewal: Your annual membership of $450 is due next month.",
    "Your Uber ride with driver John will arrive in 3 minutes. Toyota Camry - ABC 1234.",
    "Bank notification: Your credit card payment of $250.00 has been processed successfully.",
    "Hi Dad, landed safely in London. Hotel is nice. Will call you tomorrow morning.",
    "Your dentist appointment is scheduled for next Tuesday at 10 AM. Reply to confirm.",
    "Package from John Smith delivered to your front door at 2:35 PM today.",
    "Your library books are due in 3 days. You can renew them online or by calling us.",
    "Weather alert: Heavy rain expected tomorrow. Please drive carefully.",
    "Your ticket for The Metropolitan Opera on Dec 15 at 7:30 PM is confirmed.",
    "Project deadline extended to next Friday. Let me know if you have any questions.",
    "Your Netflix subscription of $15.99 will be charged on the 1st of next month.",
    "Congratulations on your promotion! Well deserved. Let's grab lunch to celebrate.",
    "Your property tax payment has been received. Receipt attached for your records.",
    "Reminder: Parent-teacher conference scheduled for Thursday at 4 PM in Room 205.",
    "Your car service appointment at Toyota dealer is confirmed for Monday at 9 AM.",
    "Book club meeting this Wednesday at 6pm. This month we're discussing '1984'."
]

# ------------------- Custom Transformer -------------------
ps = PorterStemmer()

class TransformTextWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize attributes - will be set even when unpickling
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Use fresh instances each time to avoid pickle issues
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            y = [i for i in text if i.isalnum()]
            text = [i for i in y if i not in stop_words and i not in string.punctuation]
            text = [stemmer.stem(i) for i in text]
            return " ".join(text)
        
        import pandas as pd
        if isinstance(X, list):
            X = pd.Series(X)
        return X.apply(transform_text)

# ------------------- Download NLTK Data -------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_data()

# ------------------- Load Pipeline -------------------
@st.cache_resource
def load_model():
    with open('spam_detection_pipeline_final.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

pipeline = load_model()

# ------------------- Gemini Analysis -------------------
def get_gemini_analysis(message, is_spam):
    if not GEMINI_API_KEY:
        return None
    try:
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        if is_spam:
            prompt = f"""
Our ML spam detection model has CLASSIFIED this message as SPAM.

Explain WHY this message is spam by analyzing:
1. Type of spam (phishing, scam, fraud, etc.)
2. Red flags in this message
3. What user should do next
4. Potential risks

Message: "{message}"
"""
        else:
            prompt = f"""
Our ML model has CLASSIFIED this message as LEGITIMATE.

Explain briefly why it appears safe:
1. Characteristics that indicate trustworthiness
2. Why it's likely not spam

Message: "{message}"
"""
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Unable to generate AI analysis: {str(e)}"

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
}
.stTextArea textarea {
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: 2px solid #1976D2 !important;
}
.stButton>button {
    background-color: #1976D2;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #1258a0;
    transform: translateY(-2px);
}
.result-flex {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 2rem;
    margin-top: 2rem;
}
.result-box {
    flex: 1;
    min-width: 45%;
    background: linear-gradient(135deg, #ffffff, #f3f4f6);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    color: #222;
}
.spam-header {
    font-size: 2rem;
    color: #c62828;
    font-weight: 800;
}
.safe-header {
    font-size: 2rem;
    color: #2e7d32;
    font-weight: 800;
}
.analysis-scroll {
    max-height: 450px;
    overflow-y: auto;
    background-color: #fafafa;
    border-radius: 15px;
    padding: 1rem;
    line-height: 1.7;
    border: 2px solid #2196F3;
}
.analysis-scroll::-webkit-scrollbar {
    width: 10px;
}
.analysis-scroll::-webkit-scrollbar-thumb {
    background: #2196F3;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #283E51, #485563); border-radius: 15px;'>
    <h1 style='color: white; font-size: 2.8rem;'>📧 Email Spam Detection using NLP & Machine Learning</h1>
    <p style='color: #dfe6e9; font-size: 1.2rem;'>AI-powered detection and analysis of suspicious messages</p>
</div>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spam.png", width=90)
    st.header("💡 About the App")
    st.write("""
- Detects spam using **Natural Language Processing (NLP)**  
- Classifies using **Machine Learning**  
- Provides **AI-powered Gemini explanations**
""")
    
    st.markdown("---")
    st.subheader("🧪 Try Sample Emails")
    st.write("Click below to load a random email (spam or legitimate):")
    
    if st.button("🎲 Try Random Email", use_container_width=True):
        # Combine all samples and pick one randomly
        all_samples = SPAM_SAMPLES + NOT_SPAM_SAMPLES
        st.session_state.text = random.choice(all_samples)
        st.rerun()

# ------------------- Input Area -------------------
default_text = st.session_state.get('text', '')
input_sms = st.text_area("✉️ Enter your email or SMS message:", value=default_text, height=180)

# ------------------- Buttons -------------------
col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 Analyze Message", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.text = ''
        st.rerun()

# ------------------- Prediction Logic -------------------
if analyze:
    if input_sms.strip():
        with st.spinner("Analyzing message..."):
            time.sleep(1)
            result = pipeline.predict([input_sms])[0]
            analysis = get_gemini_analysis(input_sms, result == 1)
        
        # --- Display Results in Flexbox Layout ---
        st.markdown("<div class='result-flex'>", unsafe_allow_html=True)
        
        if result == 1:
            st.markdown(f"""
            <div class='result-box'>
                <div class='spam-header'>🚨 SPAM DETECTED</div>
                <p style='color:#b71c1c;'>This message appears to be spam. Be cautious!</p>
            </div>
            <div class='result-box'>
                <h3 style='color:#1976D2;'>🧠 AI-Powered Analysis</h3>
                <div class='analysis-scroll'>{analysis.replace("\n","<br>") if analysis else "AI analysis unavailable."}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-box'>
                <div class='safe-header'>✅ LEGITIMATE MESSAGE</div>
                <p style='color:#2e7d32;'>This message appears to be safe.</p>
            </div>
            <div class='result-box'>
                <h3 style='color:#1976D2;'>🧠 AI Analysis</h3>
                <div class='analysis-scroll'>{analysis.replace("\n","<br>") if analysis else "AI analysis unavailable."}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to analyze.")

# ------------------- Footer -------------------
st.markdown("""
<hr>
<div style='text-align:center; color:#ccc;'>
    <p>📧 Built with <b>Streamlit</b> | Powered by <b>Machine Learning</b> & <b>Gemini AI</b></p>
</div>
""", unsafe_allow_html=True)
