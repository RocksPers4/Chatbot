# backend/app.py
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from routes.chat import chat_bp
from services.chatbot import ChatbotService
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

def create_app():
    app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
    CORS(app)
    app.config.from_object(Config)

    logging.info(f"Configuración de MySQL: Host={Config.MYSQL_HOST}, User={Config.MYSQL_USER}, DB={Config.MYSQL_DB}, Port={Config.MYSQL_PORT}")

    # Inicializar el chatbot
    ChatbotService.initialize()

    # Registrar el Blueprint para las rutas
    app.register_blueprint(chat_bp)

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route('/health')
    def health_check():
        is_db_connected = ChatbotService.connection is not None and ChatbotService.connection.is_connected()
        return jsonify({
            "status": "healthy" if is_db_connected else "unhealthy",
            "database_connected": is_db_connected
        }), 200 if is_db_connected else 500

    @app.route('/api/chat', methods=['POST'])
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