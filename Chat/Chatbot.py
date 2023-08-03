from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("facebook/opt-350m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

# Initialize the conversation history as a list
dialogue_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global dialogue_history

    user_message = request.form['user_message']

    # Append user message to the dialogue history
    dialogue_history.append({'user': user_message})

    # Generate a response using the entire conversation history
    input_text = ' '.join([entry['user'] for entry in dialogue_history])
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Append bot response to the dialogue history
    dialogue_history[-1]['bot'] = bot_response

    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run()
