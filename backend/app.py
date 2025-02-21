import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from backend.routes.chat import chat_bp

app = Flask(__name__)
app.register_blueprint(chat_bp)

if __name__ == '__main__':
    app.run(debug=True)

