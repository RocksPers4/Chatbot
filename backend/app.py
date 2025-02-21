# backend/app.py
from flask import Flask
from flask_cors import CORS
from routes.chat import chat_bp, chatbot
from config import Config

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Inicializar el chatbot antes de registrar las rutas
chatbot.initialize()

# Registrar el Blueprint para las rutas
app.register_blueprint(chat_bp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)  # Solo para pruebas locales
