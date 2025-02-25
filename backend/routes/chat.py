from flask import Blueprint, request, jsonify
from services.chatbot import ChatbotService

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        response = ChatbotService.get_response(message)
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
    
    feedback_response = ChatbotService.handle_feedback(feedback, last_response)
    return jsonify({"response": feedback_response})

@chat_bp.route('/clear-history', methods=['POST'])
def clear_history():
    ChatbotService.clear_history()
    return jsonify({"message": "Historial borrado con éxito"}), 200

@chat_bp.route('/health', methods=['GET'])
def healthcheck():
    return jsonify({"status": "OK"}), 200