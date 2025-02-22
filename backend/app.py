# backend/app.py
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from routes.chat import chat_bp
from services.chatbot import ChatbotService
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)

    logging.info(f"Configuración de MySQL: Host={Config.MYSQL_HOST}, User={Config.MYSQL_USER}, DB={Config.MYSQL_DB}, Port={Config.MYSQL_PORT}")

    # Inicializar el chatbot
    ChatbotService.initialize()

    # Registrar el Blueprint para las rutas
    app.register_blueprint(chat_bp)

    @app.route('/health')
    def health_check():
        is_db_connected = ChatbotService.connection is not None and ChatbotService.connection.is_connected()
        return jsonify({
            "status": "healthy" if is_db_connected else "unhealthy",
            "database_connected": is_db_connected
        }), 200 if is_db_connected else 500

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.json
        message = data.get('message', '')
        response = ChatbotService.get_response(message)
        return jsonify({"response": response})

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Error interno del servidor. Por favor, contacta al administrador."}), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host="0.0.0.0", port=port)
