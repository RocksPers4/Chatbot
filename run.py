import os
from dotenv import load_dotenv
from backend.app import app

load_dotenv()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)