# backend/app.py
from flask import Flask, jsonify
from flask_cors import CORS
from routes.chat import chat_bp, chatbot
from config import Config

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)

    # Inicializar el chatbot
    chatbot.initialize()

    # Registrar el Blueprint para las rutas
    app.register_blueprint(chat_bp)

    # Manejo de errores personalizados
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Recurso no encontrado"}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Error interno del servidor. Por favor, contacta al administrador."}), 500

    # Manejo de errores generales
    @app.errorhandler(Exception)
    def handle_exception(error):
        return jsonify({"error": "Ha ocurrido un error inesperado"}), 500

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)  # Solo para pruebas locales
