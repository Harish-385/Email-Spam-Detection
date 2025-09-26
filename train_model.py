import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Ensure data directory exists
    if not os.path.exists('data'):
        raise FileNotFoundError("Data directory not found. Please make sure you're running the script from the correct directory.")

    # Ensure model directory exists
    if not os.path.exists('model'):
        os.makedirs('model')
        logging.info("Created model directory")

    # Load dataset
    logging.info("Loading data from spam.csv...")
    if not os.path.exists('data/spam.csv'):
        raise FileNotFoundError("spam.csv not found in data directory")
    
    df = pd.read_csv('data/spam.csv', encoding='latin-1', sep='\t', names=['label', 'message'])
    df = df.dropna()
    df = df[df['label'].isin(['ham', 'spam'])]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    logging.info(f"Data loaded successfully. Shape: {df.shape}")

    # Split dataset
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Vectorization with TF-IDF + n-grams
    logging.info("Vectorizing text data with TF-IDF (1-3 grams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logging.info("Text vectorization completed")

    # Train Naive Bayes model
    logging.info("Training MultinomialNB model with alpha=0.1...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)
    logging.info("Model training completed")

    # Evaluate model
    train_accuracy = model.score(X_train_vec, y_train)
    test_accuracy = model.score(X_test_vec, y_test)
    logging.info(f"Training accuracy: {train_accuracy:.4f}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}")

    y_pred = model.predict(X_test_vec)
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # Save model and vectorizer
    logging.info("Saving model and vectorizer...")
    joblib.dump(model, 'model/spam_classifier.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    logging.info("Model and vectorizer saved successfully")

    # --- Test function for new messages ---
    def predict_message(message):
        vec = vectorizer.transform([message])
        pred = model.predict(vec)[0]
        label = "Ham" if pred == 0 else "Spam"
        prob = model.predict_proba(vec)[0][pred]
        return f"Prediction: {label} (Confidence: {prob:.2f})"

    # Example test with your ham message
    test_msg = "Can you please update the project timeline by end of day?"
    print(predict_message(test_msg))

except FileNotFoundError as e:
    logging.error(f"File not found: {str(e)}")
    raise
except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    raise
