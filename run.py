import os

port = os.getenv("PORT", "8080")
print(f"🚀 Iniciando en el puerto: {port}")  # <-- Esto imprimirá el puerto en los logs de Railway

from backend.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(port))
