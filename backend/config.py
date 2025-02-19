# backend/config.py
import os

class Config:
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/chatbot'
    # Añade aquí otras configuraciones necesarias