# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()
#mysql://root:tQnPxOGTPXzPPRjVexBFZytraYuzuDfF@shuttle.proxy.rlwy.net:55939/railway
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MYSQL_HOST = os.environ.get('MYSQLHOST', 'shuttle.proxy.rlwy.net')
    MYSQL_USER = os.environ.get('MYSQLUSER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQLPASSWORD', 'tQnPxOGTPXzPPRjVexBFZytraYuzuDfF')
    MYSQL_DB = os.environ.get('MYSQLDATABASE', 'railway')
    MYSQL_PORT = int(os.environ.get('MYSQLPORT', 55939))
    
    @classmethod
    def get_mysql_uri(cls):
        return f"mysql+mysqlconnector://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DB}"