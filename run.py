import os
from dotenv import load_dotenv
from backend.app import app

load_dotenv()

if __name__ == '__main__':
    port = int(os.getenv("PORT", "8080"))  # Asegurar que el puerto por defecto sea el correcto
    app.run(host='0.0.0.0', port=port)
