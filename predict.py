import joblib

model = joblib.load('model/spam_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def predict_email(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "Spam" if prediction[0] == 1 else "Ham"
