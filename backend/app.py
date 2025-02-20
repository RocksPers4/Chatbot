# backend/app.py
import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from routes.chat import chat_bp, chatbot
from config import Config

app = Flask(__name__, static_folder='../frontend/build')
CORS(app)
app.config.from_object(Config)

app.register_blueprint(chat_bp)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')


def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    chatbot.initialize()  # Inicializa el chatbot si es necesario
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)