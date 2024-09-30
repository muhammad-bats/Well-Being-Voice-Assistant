from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask('AI Chatbot')

# Loading GPT-2 model and tokenizer
text_model = AutoModelForSequenceClassification.from_pretrained("AlexKay/xlm-roberta-base-finetuned-CounselChat")
text_tokenizer = AutoTokenizer.from_pretrained("AlexKay/xlm-roberta-base-finetuned-CounselChat")

def generate_response(user_input):

    text_tokenizer.pad_token = text_tokenizer.eos_token # Manual padding
    # Encoding the user input and generate the attention mask
    inputs = text_tokenizer.encode_plus(
        user_input, 
        return_tensors='pt',  # To Return PyTorch tensors
        padding=True,         # Pad the inputs to max length
        truncation=True,      # Truncate inputs to fit within the model's max length
        max_length=512        # Maximum length for the input sequence
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate response using GPT-2[selected model]
    with torch.no_grad():
        chatbot_output = text_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,         # Control randomness
            top_k=50,                # Consider only top 50 tokens
            top_p=0.9,               # Use nucleus sampling
            no_repeat_ngram_size=2    # Prevent repetition of bigrams
        )
    
    response = text_tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
    # Strip out any repetition of the user's input from the response
    response = response.replace(user_input, '').strip()
    
    return response

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.get_json()
    user_input = data.get('message')
    response = generate_response(user_input)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
