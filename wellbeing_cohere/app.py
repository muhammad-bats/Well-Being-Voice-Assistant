from flask import Flask, request, jsonify, render_template
import cohere

app = Flask("AI Chatbot")

co = cohere.Client("ly6VpnzTXQvxe45hV0AOVCixO0L01DliJqKHGTAr")

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.get_json()
    user_message = data.get('message')

    response = co.generate(
        model="command-xlarge-nightly",
        prompt= user_message,
        max_tokens=300
    )
    bot_response = response.generations[0].text

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
