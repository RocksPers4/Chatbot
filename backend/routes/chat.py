from flask import Blueprint, request, jsonify
from services.chatbot import ChatbotService  # Update this import to match your actual file structure

chat_bp = Blueprint('chat', __name__)
chatbot = ChatbotService()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        response = chatbot.get_response(message)  # Removed user_id parameter
        # If you still want to save the conversation, you can do it here without a user_id
        # For example: Conversation.save(message, response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({'response': response}), 200

@chat_bp.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback = data.get('feedback')
    last_response = data.get('last_response')
    
    if not feedback or not last_response:
        return jsonify({"error": "Se requiere feedback y last_response"}), 400
    
    feedback_response = chatbot.handle_feedback(feedback, last_response)
    return jsonify({"response": feedback_response})