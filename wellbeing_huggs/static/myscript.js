//const recordButton = document.getElementById('voice-button');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');

// Initialize Speech Recognition
function recordUser() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = function(event) {
        const message = event.results[0][0].transcript;
        addMessage('You', message, 'user');
        sendToBackend(message);
    };

    recognition.onerror = function(event) {
        alert('Error occurred in recognition: ' + event.error);
    };
}

function sendText() {
    const message = userInput.value;
    if (message.trim()){
        addMessage("You", message, "user");
        sendToBackend(message);
        userInput.value = '';
    }
}

function sendToBackend(message) {
    fetch('/process_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'message': message })
    })
    .then(response => response.json())
    .then(data => {
        addMessage('Bot:', data.response, 'bot');
    });
}

function addMessage(sender, message, className) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${className}`;
    if (sender === 'Bot:') {
        console.log(sender)
        messageElement.innerHTML = `<i class="fa-solid fa-robot"></i> ${message}`;
    }  else {
        messageElement.innerHTML = `${message}`;
    }  
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the bottom
}