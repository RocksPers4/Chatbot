import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from routes.chat import chat_bp, chatbot
from werkzeug.utils import secure_filename
from config import Config

# Crear la aplicación Flask
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)  # Habilitar CORS
app.config.from_object(Config)

# Registrar el blueprint del chatbot
app.register_blueprint(chat_bp)

# Ruta para servir los archivos estáticos del frontend (React)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # Enviar archivos estáticos
    safe_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(safe_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Asegúrate de que el chatbot se inicializa correctamente
chatbot.initialize()  # Inicializa el chatbot si es necesario

# Eliminar app.run() ya que usaremos gunicorn para la producción
if __name__ == '__main__':
    # En producción gunicorn se encargará de iniciar la app
    pass
