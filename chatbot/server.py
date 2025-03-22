from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # Limit conversation history
    if len(conversation_history) > 6:
        conversation_history.pop(0)

    conversation_history.append(input_text)

    # Tokenize input text only
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=100)

    # Generate response with proper parameters
    outputs = model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run(debug=True)
