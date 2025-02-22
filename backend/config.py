# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MYSQL_HOST = os.environ.get('MYSQL_HOST') or 'localhost'
    MYSQL_USER = os.environ.get('MYSQL_USER') or 'root'
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD') or 'zoe1506'
    MYSQL_DB = os.environ.get('MYSQL_DB') or 'espoch_chatbot'
    MYSQL_PORT = int(os.environ.get('MYSQL_PORT') or 3306)