from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['user_message']
    # Add chatbot logic here to generate a response based on user_message
    # For this example, we'll just echo the user's message as the bot's response
    bot_response = user_message
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
