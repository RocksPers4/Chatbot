import os
from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))  # Usa 4000 si no hay otra variable de entorno
    app.run(host="0.0.0.0", port=port)