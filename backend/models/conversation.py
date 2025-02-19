from pymongo import MongoClient
from datetime import datetime
from config import Config

client = MongoClient(Config.MONGO_URI)
db = client.get_database()

conversations = db.conversations

class Conversation:
    @staticmethod
    def save(user_id, message, response):
        conversations.insert_one({
            'user_id': user_id,
            'message': message,
            'response': response,
            'timestamp': datetime.utcnow()
        })

    @staticmethod
    def get_history(user_id):
        return list(conversations.find({'user_id': user_id}).sort('timestamp', -1))