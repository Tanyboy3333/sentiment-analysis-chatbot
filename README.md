✨Sentiment Analysis Chatbot



This project is a Sentiment Analysis Chatbot that uses VADER (lexicon-based) and a Logistic Regression (machine learning-based) model for robust sentiment prediction.
It cleans text, analyzes with both models, and gives a final combined prediction.



Live Demo 🚀: https://huggingface.co/spaces/Tanveer3333/Sentiment_Chatbot 



📚 Project Workflow
1. Data Preparation
Collected and labeled textual data for sentiment (positive, negative, neutral).

Preprocessed text using:

URL removal

Special character removal

Lowercasing

Stopword removal

Lemmatization

2. Model Training
Vectorized text data using TF-IDF vectorizer.

Trained a Logistic Regression model on the TF-IDF features.

Achieved high accuracy on the validation set.

Saved the trained model and vectorizer using joblib.

3. Building the Chatbot App
Built a Gradio interface with:

VADER sentiment analysis

Logistic Regression model prediction

Smart combination of both predictions

Cleaned and processed user input text in real-time.

4. Deployment
Created a lightweight and portable app using Gradio.

Deployed the app on Hugging Face Spaces for free, making it accessible globally.


🛠 Technologies Used
Python 🐍

NLTK (Natural Language Toolkit)

Scikit-learn

Gradio (for creating the UI)

Hugging Face Spaces (for deployment)

Joblib (for model serialization)

🧠 Features
Real-time sentiment prediction.
(For any Input the Output ill be of the form 1:Positive Sentiment, 0:Neutral Sentiment, -1:Negative Sentiment)

Hybrid approach combining VADER and Machine Learning.

✨ Future Enhancements
Fine-tune a transformer model (e.g., BERT) for better accuracy.

Add support for multilingual sentiment analysis.

Create a more advanced conversational agent.
