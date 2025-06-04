import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- Page Config (MUST BE FIRST) ---
st.set_page_config(
    page_title="AI Fake News Detector", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer with error handling
@st.cache_resource
def load_models():
    try:
        # Check if model files exist
        if not os.path.exists("fake_news_model.pkl"):
            st.error("❌ Model file 'fake_news_model.pkl' not found. Please ensure the model file is in the same directory.")
            return None, None
        
        if not os.path.exists("tfidf_vectorizer.pkl"):
            st.error("❌ Vectorizer file 'tfidf_vectorizer.pkl' not found. Please ensure the vectorizer file is in the same directory.")
            return None, None
            
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Initialize models
model, vectorizer = load_models()

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
    }
    
    .real-news {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left-color: #28a745;
        color: #155724;
    }
    
    .fake-news {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-top: 3px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #856404;
    }
    
    .info-box {
        background: linear-gradient(135deg, #cce7ff, #b8daff);
        border: 1px solid #0066cc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #004085;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def analyze_text_features(text):
    """Analyze text and return various statistics"""
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    
    # Count exclamation marks and question marks
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Count uppercase words (potential indicator of sensationalism)
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    return {
        "Character Count": len(text),
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Average Word Length": round(avg_word_length, 1),
        "Average Sentence Length": round(avg_sentence_length, 1),
        "Exclamation Marks": exclamation_count,
        "Question Marks": question_count,
        "Uppercase Words": uppercase_words
    }

def get_prediction_explanation(prediction, confidence):
    """Generate explanation based on prediction and confidence"""
    if prediction == 1:  # Real news
        if confidence > 90:
            return "The model is highly confident this is legitimate news content."
        elif confidence > 75:
            return "The model indicates this is likely authentic news with good confidence."
        else:
            return "The model suggests this is real news, but with moderate confidence. Consider additional verification."
    else:  # Fake news
        if confidence > 90:
            return "The model strongly indicates this content has characteristics of misinformation."
        elif confidence > 75:
            return "The model detected patterns commonly associated with fake news."
        else:
            return "The model suggests this might be fake news, but with moderate confidence. Further verification recommended."

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.title("🧭 Navigation")  
    st.markdown("Navigate through the app sections:")
    st.markdown('</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Select Page", 
        ["🔍 Detect Fake News", "📖 About", "⚙️ How It Works", "📊 Examples"],
        index=0
    )
    
    # Add model status indicator
    st.markdown("---")
    if model is not None and vectorizer is not None:
        st.success("✅ Models loaded successfully")
    else:
        st.error("❌ Models not available")
    
    # Add quick tips
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - 📝 Paste complete articles for best results
    - 📏 Minimum 50 words recommended
    - 🔍 Check multiple sources for verification
    - 🎯 Consider context and source credibility
    - ⚡ Real-time analysis in under 1 second
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Detection Page ---
if page == "🔍 Detect Fake News":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI-Powered Fake News Detector</h1>
        <p>Analyze news articles using advanced machine learning to detect misinformation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are loaded
    if model is None or vectorizer is None:
        st.error("⚠️ Cannot perform analysis. Please ensure model files are available.")
        st.info("Required files: `fake_news_model.pkl` and `tfidf_vectorizer.pkl`")
        st.stop()
    
    st.markdown("### ✍️ Enter News Article")
    user_input = st.text_area(
        "",
        height=300, 
        placeholder="""Paste your news article here...

Example: "Scientists have discovered a new planet in our solar system that could support human life. The planet, located between Mars and Jupiter, has been named 'Terra Nova' and shows signs of water and breathable atmosphere..."
        """,
        help="Paste the full text of a news article you want to analyze (minimum 50 words recommended)"
    )
    
    # Quick word count display
    if user_input:
        word_count = len(user_input.split())
        if word_count < 20:
            st.warning(f"⚠️ Current word count: {word_count}. For better accuracy, please provide at least 50 words.")
        else:
            st.info(f"📝 Word count: {word_count}")
        
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_confidence = st.checkbox("Show Confidence Score", value=True)
    with col_opt2:
        show_analysis = st.checkbox("Show Text Analysis", value=True)
    with col_opt3:
        show_explanation = st.checkbox("Show Explanation", value=True)
        
    # Predict button
    predict_button = st.button(
        "🔍 Analyze Article", 
        type="primary",
        use_container_width=True
    )
    
    # Prediction logic
    if predict_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some news text to analyze.")
        elif len(user_input.strip()) < 20:
            st.warning("⚠️ Please enter a longer text (at least 50 words) for better accuracy.")
        else:
            # Show loading spinner
            with st.spinner('🤖 AI is analyzing the article...'):
                start_time = time.time()
                
                try:
                    # Make prediction
                    input_vec = vectorizer.transform([user_input])
                    prediction = model.predict(input_vec)[0]
                    probabilities = model.predict_proba(input_vec)[0]
                    confidence = probabilities[prediction] * 100
                    
                    processing_time = time.time() - start_time
                    
                    # Display results
                    st.markdown("## 📊 Analysis Results")
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-card real-news">
                            <h2>✅ Likely REAL News</h2>
                            <p>Our AI model indicates this article appears to be legitimate news content.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        result_emoji = "✅"
                        result_color = "green"
                    else:
                        st.markdown(f"""
                        <div class="prediction-card fake-news">
                            <h2>⚠️ Potentially FAKE News</h2>
                            <p>Our AI model has detected patterns commonly associated with misinformation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        result_emoji = "⚠️"
                        result_color = "red"
                    
                    # Show explanation if enabled
                    if show_explanation:
                        explanation = get_prediction_explanation(prediction, confidence)
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>🤖 AI Explanation:</strong><br/>
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence and basic stats
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    
                    with col_res1:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.metric("Word Count", len(user_input.split()))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_res3:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_res4:
                        reliability = "High" if confidence > 85 else "Medium" if confidence > 70 else "Low"
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.metric("Reliability", reliability)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if show_confidence:
                        st.markdown("### 📈 Confidence Breakdown")
                        
                        # Create confidence visualization
                        conf_data = pd.DataFrame({
                            'Prediction': ['Real News', 'Fake News'],
                            'Probability': [probabilities[1]*100, probabilities[0]*100]
                        })
                        
                        st.bar_chart(conf_data.set_index('Prediction'))
                        
                        # Additional confidence metrics
                        col_conf1, col_conf2 = st.columns(2)
                        with col_conf1:
                            st.metric("Real News Probability", f"{probabilities[1]*100:.1f}%")
                        with col_conf2:
                            st.metric("Fake News Probability", f"{probabilities[0]*100:.1f}%")
                    
                    if show_analysis:
                        st.markdown("### 🔍 Detailed Text Analysis")
                        
                        text_stats = analyze_text_features(user_input)
                        
                        # Display stats in a grid
                        cols = st.columns(4)
                        for i, (key, value) in enumerate(text_stats.items()):
                            with cols[i % 4]:
                                st.metric(key, value)
                    
                    # Disclaimer
                    st.markdown("---")
                    st.markdown("""
                    <div class="warning-box">
                        <strong>⚠️ Important Disclaimer:</strong><br/>
                        This tool provides AI-based analysis and should not be the sole source for determining news authenticity. 
                        Always verify important information through multiple reliable sources and use critical thinking when evaluating news content.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
                    st.info("Please try again with different text or contact support if the issue persists.")

# --- Examples Page ---
elif page == "📊 Examples":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Examples & Test Cases</h1>
        <p>Try these sample texts to see how the detector works</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧪 Sample Articles for Testing")
    
    examples = {
        "Real News Example": """
        The Federal Reserve announced today that it will maintain interest rates at their current level following a two-day meeting of the Federal Open Market Committee. 
        Fed Chair Jerome Powell cited ongoing economic uncertainty and inflation concerns as key factors in the decision. 
        The central bank's statement noted that while employment levels have shown improvement, 
        officials remain cautious about the pace of economic recovery. Market analysts had widely anticipated this decision, 
        with most major financial institutions predicting the Fed would hold rates steady. 
        The announcement comes amid continued monitoring of global economic conditions and domestic fiscal policy developments.
        """,
        
        "Suspicious Content Example": """
        SHOCKING: Scientists have discovered that drinking this common household item can extend your life by 50 years! 
        Doctors HATE this one simple trick that pharmaceutical companies don't want you to know! 
        This miracle cure has been hidden from the public for decades, but now the truth is finally revealed! 
        Click here to learn the secret that will change your life forever! 
        Thousands of people are already using this method and seeing incredible results! 
        Don't let Big Pharma control your health - take control today!
        """,
        
        "Neutral Example": """
        Local weather services report that tomorrow's forecast includes partly cloudy skies with temperatures reaching a high of 75 degrees Fahrenheit. 
        Residents can expect light winds from the southwest at approximately 10 mph. 
        There is a 20% chance of scattered showers in the late afternoon. 
        The National Weather Service advises that conditions are generally favorable for outdoor activities. 
        Weekend weather patterns suggest continued mild temperatures with similar conditions expected through Sunday.
        """
    }
    
    for title, content in examples.items():
        with st.expander(f"📄 {title}"):
            st.text_area("Content:", content, height=150, key=f"example_{title}")
            if st.button(f"Test this example", key=f"test_{title}"):
                st.info("Copy the text above and paste it into the main detection page to analyze!")
    
    st.markdown("### 📝 Testing Tips")
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 What Makes Good Test Content:</h4>
        <ul>
            <li><strong>Length:</strong> Articles with 100+ words provide better accuracy</li>
            <li><strong>Structure:</strong> Complete sentences and paragraphs work best</li>
            <li><strong>Content:</strong> News-style writing produces most reliable results</li>
            <li><strong>Language:</strong> Clear, grammatically correct English is optimal</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- How It Works Page ---
elif page == "⚙️ How It Works":
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ How It Works</h1>
        <p>Understanding the technology behind fake news detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧠 Machine Learning Pipeline")
    
    steps = [
        ("📝 Text Preprocessing", "Clean and normalize the input text, remove special characters, handle punctuation, and prepare text for analysis"),
        ("🔤 TF-IDF Vectorization", "Convert text into numerical features using Term Frequency-Inverse Document Frequency, capturing word importance"),
        ("🤖 Logistic Regression", "Apply trained classification model to analyze text patterns and predict authenticity based on learned features"),
        ("📊 Confidence Scoring", "Calculate prediction confidence using model probabilities and provide reliability metrics")
    ]
    
    for i, (title, description) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="feature-card">
            <h3>Step {i}: {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Model Performance & Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Training Dataset</h4>
            <ul>
                <li>20,000+ news articles</li>
                <li>Balanced real/fake distribution</li>
                <li>Multiple news sources</li>
                <li>Cross-validated results</li>
                <li>Diverse topic coverage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚙️ Technical Features</h4>
            <ul>
                <li>TF-IDF feature extraction</li>
                <li>Logistic regression classifier</li>
                <li>~94% accuracy rate</li>
                <li>Real-time prediction</li>
                <li>Confidence scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 What the Model Analyzes")
    
    analysis_features = [
        ("Word Patterns", "Frequency and distribution of specific words and phrases commonly associated with reliable or unreliable sources"),
        ("Writing Style", "Sentence structure, grammar patterns, and linguistic markers that may indicate authenticity"),
        ("Content Structure", "Organization of information, presence of quotes, citations, and journalistic conventions"),
        ("Sensationalism Indicators", "Excessive use of capital letters, exclamation marks, and emotionally charged language")
    ]
    
    for feature, description in analysis_features:
        st.markdown(f"""
        <div class="feature-card">
            <h4>🔸 {feature}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

# --- About Page ---
elif page == "📖 About":
    st.markdown("""
    <div class="main-header">
        <h1>📖 About This Application</h1>
        <p>AI-powered solution for combating misinformation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Mission")
    st.markdown("""
    In today's digital age, misinformation spreads rapidly across social media and news platforms. 
    This AI-powered tool helps users quickly identify potentially fake news articles using advanced 
    machine learning techniques. Our goal is to promote digital literacy and provide users with 
    tools to make more informed decisions about the content they consume and share.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛠️ Technology Stack</h3>
            <ul>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>ML Model:</strong> Logistic Regression</li>
                <li><strong>Feature Extraction:</strong> TF-IDF Vectorization</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                <li><strong>Model Persistence:</strong> Joblib</li>
                <li><strong>Deployment:</strong> Python 3.8+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Model Statistics</h3>
            <ul>
                <li><strong>Accuracy:</strong> ~94.2%</li>
                <li><strong>Training Data:</strong> 20,000+ articles</li>
                <li><strong>Processing Time:</strong> < 1 second</li>
                <li><strong>Languages:</strong> English</li>
                <li><strong>Model Type:</strong> Supervised Learning</li>
                <li><strong>Last Updated:</strong> January 2025</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 Research & Development")
    st.markdown("""
    This application is based on established research in natural language processing and machine learning 
    for misinformation detection. The model uses TF-IDF (Term Frequency-Inverse Document Frequency) 
    vectorization combined with logistic regression, a proven approach for text classification tasks.
    """)
    
    st.markdown("### 👨‍💻 Developer")
    st.markdown("""
    **Muhammad Umar Tariq**  
    AI/ML Engineer passionate about using technology to combat misinformation and promote digital literacy.
    
    *Created with ❤️ and Python*
    """)
    
    st.markdown("### ⚠️ Important Limitations")
    st.markdown("""
    <div class="warning-box">
        <strong>Please Note:</strong><br/>
        <ul>
            <li>This tool is designed to assist in identifying potentially false information, but should not be the only method used to verify news authenticity</li>
            <li>Always cross-reference important information with multiple reliable sources</li>
            <li>Use critical thinking when evaluating news content</li>
            <li>The model may not perform equally well on all types of content or topics</li>
            <li>Continuous updates and improvements are needed to maintain effectiveness</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📞 Contact & Support")
    st.info("""
    For questions, feedback, or technical issues, please reach out through appropriate channels.
    Your feedback helps improve the accuracy and effectiveness of this tool.
    """)

# --- Footer ---
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 8px; margin-top: 2rem;'>
    <strong>🕒 Last updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
    <strong>Version:</strong> 1.1
</div>
""", unsafe_allow_html=True)