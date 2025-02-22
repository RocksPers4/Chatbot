# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MYSQL_HOST = os.environ.get('MYSQLHOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQLUSER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQLPASSWORD', 'zoe1506')
    MYSQL_DB = os.environ.get('MYSQLDATABASE', 'espoch_chatbot')
    MYSQL_PORT = int(os.environ.get('MYSQLPORT', 3306))

    @classmethod
    def get_mysql_uri(cls):
        return f"mysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DB}"