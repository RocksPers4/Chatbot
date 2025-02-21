import os
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from backend.routes.chat import chat_bp, chatbot

def create_app():
    app = Flask(__name__, static_folder='../frontend/build')
    CORS(app)

    app.register_blueprint(chat_bp)

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error=str(e)), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify(error=str(e)), 500

    return app

app = create_app()

if __name__ == '__main__':
    chatbot.initialize()  # Inicializa el chatbot antes de arrancar la app
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port)
