# backend/app.py
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
    app.run(debug=True)
