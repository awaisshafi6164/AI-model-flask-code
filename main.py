import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
import pickle
import spacy
import os
#below imports are for summary_text model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)

# Get the absolute path to the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "emotion_classifier_model.pkl")

# Load the saved model from the file
with open(model_path, "rb") as f:
    clf = pickle.load(f)

# Load spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Preprocess function
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        else:
            filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

# Emotion mapping dictionary
emotion_mapping = {0: 'happy', 1: 'sad', 2: 'worst', 3: 'worst', 4: 'fantastic', 5: 'fine'}

# Predict emotion from text
def predict_emotion_from_text(input_text):
    # Preprocess the input text
    processed_text = preprocess(input_text)

    # Predict emotion using the loaded model
    emotion_label = clf.predict([processed_text])[0]

    # Map the predicted label to emotion
    predicted_emotion = emotion_mapping[emotion_label]

    return predicted_emotion

# Route for predicting emotion
@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    data = request.get_json()
    input_text = data['text']
    predicted_emotion = predict_emotion_from_text(input_text)
    return jsonify({'predicted_emotion': predicted_emotion})


#code of summary text model
model_path = "simplet5-epoch-19-train-loss-0.7495-val-loss-3.1781/"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Route for summarizing text
@app.route('/summarize-text', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data['text']
    preprocess_text = input_text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=512,
                                 early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({'summary': output})

# # Route for summarizing text
# @app.route('/summarize-text', methods=['POST'])
# def summarize_text():
#     data = request.get_json()
#     input_text = data['text']
#     processed_text = preprocess(input_text)
#     return jsonify({'summary': processed_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
