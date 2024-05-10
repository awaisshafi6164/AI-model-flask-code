import os

def get_file_size_mb(file_path):
  # Get the size of the file in bytes
  size_bytes = os.path.getsize(file_path)
  # Convert bytes to megabytes
  size_mb = size_bytes / (1024 * 1024)
  return size_mb

# Example usage:
file_path = "emotion_classifier_model.pkl"  # Replace with the path to your file
file_size_mb = get_file_size_mb(file_path)
print("File size:", file_size_mb, "MB")

import pickle
# Load the saved model from the file
with open("emotion_classifier_model.pkl", "rb") as f:
  clf = pickle.load(f)

import spacy
nlp = spacy.load("en_core_web_sm")

#preprocees code
def preprocess(text):
  doc = nlp(text)
  filtered_tokens = []
  for token in doc:
    if token.is_stop or token.is_punct:
      continue
    else:
      filtered_tokens.append(token.lemma_)
  return " ".join(filtered_tokens)

def predict_emotion_from_text(input_text):
  # Preprocess the input text
  processed_text = preprocess(input_text)

  # Predict emotion using the loaded model
  emotion_label = clf.predict([processed_text])[0]
                                
  # Map the predicted label to emotion
  emotion_mapping = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'love', 5: 'surprise'}
  predicted_emotion = emotion_mapping[emotion_label]

  return predicted_emotion

def summary_text(input_text):
  processed_text = preprocess(input_text)
  return processed_text

# Example usage
user_input = "To unwind, I took a long walk in the evening, appreciating the beauty of nature and the calming effect it had on my mood. As the day comes to an end, I find myself reflecting on the mix of emotions experienced."
predicted_emotion = predict_emotion_from_text(user_input)
summary = summary_text(user_input)
print(f"Predicted Emotion: {predicted_emotion}")
print(f"Summary Text: {summary}")