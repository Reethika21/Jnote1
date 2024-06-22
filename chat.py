import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


with open('datasets/intens.json') as json_file:
    intents_data = json.load(json_file)


patterns = []
intent_labels = []
responses = {}
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        intent_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']


label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('datasets/label_encoder_functional.npy')
vectorizer = CountVectorizer()
vectorizer.vocabulary_ = np.load('datasets/vectorizer_functional.npy', allow_pickle=True).item()


model = load_model('datasets/intent_classification_model_functional.h5')

def classify_intent(user_input):

    input_text = vectorizer.transform([user_input])
    input_text = input_text.toarray()
    predictions = model.predict(input_text)
    predicted_index = np.argmax(predictions)
    predicted_intent = label_encoder.inverse_transform([predicted_index])[0]


    return predicted_intent

def get_response(intent):
    return np.random.choice(responses[intent])

