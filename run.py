import os
from backend.app import app
from flask import Flask

app = Flask(__name__)

if __name__ == '__main__':
    port = os.environ.get("PORT", 5050)  # Si no existe PORT, usa 5000
    app.run(host="0.0.0.0", port=int(port))