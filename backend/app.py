# backend/app.py
import os
from flask import Flask
from flask_cors import CORS
from routes.chat import chat_bp, chatbot
from config import Config

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

app.register_blueprint(chat_bp)

if __name__ == '__main__':
    chatbot.initialize()  # Inicializa el chatbot si es necesario
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)