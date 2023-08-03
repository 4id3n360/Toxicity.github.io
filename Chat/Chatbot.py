from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("facebook/opt-350m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['user_message']

    # Encode the user message and generate a response
    input_ids = tokenizer.encode(user_message, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=0.7)
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run()
