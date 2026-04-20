from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)

MODEL_NAME = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

user_sessions = {}



def get_user_history(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    return user_sessions[user_id]


def build_prompt(history, user_input):
    """
    Build contextual prompt using last few turns.
    Keeps original behavior but adds memory.
    """
    context = " ".join(history[-4:])  # last 4 messages
    prompt = f"{context} {user_input}".strip()
    return prompt


def generate_response(prompt):
    """
    Generate response using model with controlled parameters.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response




@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()

        # Default user (can be extended later)
        user_id = data.get("user_id", "default_user")
        user_input = data.get("prompt", "").strip()

        if not user_input:
            return jsonify({
                "status": "error",
                "message": "Empty prompt"
            }), 400

        history = get_user_history(user_id)

        # Limit history size
        if len(history) > 6:
            history.pop(0)

        history.append(user_input)

        # Build prompt with context
        prompt = build_prompt(history, user_input)

        # Generate response
        response = generate_response(prompt)

        history.append(response)

        return jsonify({
            "status": "success",
            "response": response
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

s
if __name__ == '__main__':
    app.run(debug=True)
