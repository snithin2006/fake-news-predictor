from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('/Users/nithinsivakumar/dev/fake-news-visualizer/model/fake_news_bert_tokenizer')
model = BertForSequenceClassification.from_pretrained('/Users/nithinsivakumar/dev/fake-news-visualizer/model/fake_news_bert_model')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the incoming request
    data = request.get_json()
    text = data.get('text')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities (e.g., using softmax)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # Return the predicted class and probabilities as a JSON response
    response = {
        'predicted_class': predicted_class,
        'probabilities': probabilities.tolist()
    }

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)