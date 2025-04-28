import gradio as gr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import re
import joblib


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

import os

model_dir = os.path.dirname(os.path.abspath(__file__))
loaded_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
loaded_model = joblib.load(os.path.join(model_dir, "sentiment_model.pkl"))



sent_analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def analyze_vader_sentiment(text):
    scores = sent_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.2:
        return "positive"
    elif scores['compound'] <= -0.2:
        return "negative"
    else:
        return "neutral"


def combined_sentiment(vader_pred, ml_pred):
    if vader_pred == ml_pred:
        return vader_pred
    else:
        return ml_pred


def predict_sentiment(user_input):
    cleaned_text = clean_text(user_input)
    vader_prediction = analyze_vader_sentiment(cleaned_text)

    # Vectorize cleaned input text
    input_vector = loaded_vectorizer.transform([cleaned_text]).toarray()
    ml_prediction = loaded_model.predict(input_vector)[0]  # Predict using logistic regression model

    final_prediction = combined_sentiment(vader_prediction, ml_prediction)
    return final_prediction


def chatbot_interface(user_input):
    response = predict_sentiment(user_input)
    return f"Predicted Sentiment: {response}"

interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=5, placeholder="Enter your comment..."),
    outputs=gr.Textbox(),
    title="Sentiment Analysis Chatbot",
    description="This chatbot uses both VADER and Logistic Regression to predict sentiment."
)

# Launch the Gradio app
interface.launch()
