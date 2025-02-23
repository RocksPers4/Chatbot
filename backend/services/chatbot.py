import os
import random
import logging
import mysql.connector
from mysql.connector import Error
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import Config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ChatbotService:
    connection = None
    tokenizer = None
    model = None
    qa_pipeline = None
    vectorizer = None
    conversation_history = []
    stop_words = set(stopwords.words('spanish'))

    @classmethod
    def initialize(cls):
        """Inicializa la conexión a la base de datos."""
        try:
            cls.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                port=Config.MYSQL_PORT
            )
            if cls.connection.is_connected():
                logging.info("Conectado a MySQL correctamente.")

            cls.vectorizer = TfidfVectorizer(stop_words=list(cls.stop_words))
            cls.vectorizer.fit(cls.get_all_intents())

        except mysql.connector.Error as e:
            logging.error(f"Error al conectar a MySQL: {e}")
            raise

    @classmethod
    def get_all_intents(cls):
        """Obtiene todas las preguntas de intents de la base de datos."""
        try:
            cursor = cls.connection.cursor()
            query = """
            SELECT pregunta FROM preguntas_beca
            UNION
            SELECT pregunta FROM preguntas_ayudas
            UNION
            SELECT pregunta FROM preguntas_saludo
            """
            cursor.execute(query)
            intents = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return intents
        except Exception as e:
            logging.error(f"Error al obtener intents: {e}")
            return []

    @classmethod
    def match_intent(cls, message):
        """Busca un intent correspondiente al mensaje en la base de datos."""
        try:
            cursor = cls.connection.cursor(dictionary=True)
            query = """
            SELECT respuesta FROM (
                SELECT rb.respuesta FROM preguntas_beca pb 
                JOIN intents_beca ib ON pb.intent_beca_id = ib.id 
                JOIN respuestas_beca rb ON rb.intent_beca_id = ib.id 
                WHERE LOWER(pb.pregunta) LIKE %s
                UNION
                SELECT ra.respuesta FROM preguntas_ayudas pa 
                JOIN intents_ayudas ia ON pa.intent_id = ia.id 
                JOIN respuestas_ayudas ra ON ra.intent_id = ia.id 
                WHERE LOWER(pa.pregunta) LIKE %s
                UNION
                SELECT rs.respuesta FROM preguntas_saludo ps 
                JOIN intents_saludo isa ON ps.intent_saludo_id = isa.id 
                JOIN respuestas_saludo rs ON rs.intent_saludo_id = isa.id 
                WHERE LOWER(ps.pregunta) LIKE %s
            ) AS respuestas LIMIT 1
            """
            cursor.execute(query, (f"%{message.lower()}%", f"%{message.lower()}%", f"%{message.lower()}%"))
            result = cursor.fetchone()
            cursor.close()
            return result["respuesta"] if result else None
        except Exception as e:
            logging.error(f"Error al buscar intent: {e}")
            return None

    @classmethod
    def load_models(cls):
        """Carga el modelo de IA solo si es necesario."""
        if cls.tokenizer is None or cls.model is None:
            # Usamos un modelo más pequeño: DistilBERT multilingüe
            cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            cls.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")
            cls.qa_pipeline = pipeline("question-answering", model=cls.model, tokenizer=cls.tokenizer)

    @classmethod
    def get_bert_response(cls, context, question):
        """Genera una respuesta con DistilBERT."""
        cls.load_models()
        with torch.no_grad():
            max_length = 512
            context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
            
            best_answer = ""
            best_score = 0
            
            for chunk in context_chunks:
                result = cls.qa_pipeline(question=question, context=chunk)
                if result['score'] > best_score:
                    best_answer = result['answer']
                    best_score = result['score']

            return best_answer.strip() if best_answer else "No tengo suficiente información para responder."

    @classmethod
    def get_response(cls, message):
        """Genera la respuesta al mensaje del usuario."""
        if cls.connection is None or not cls.connection.is_connected():
            cls.initialize()

        cls.conversation_history.append({"role": "user", "content": message})

        if len(cls.conversation_history) > 5:
            cls.conversation_history.pop(0)

        intent_response = cls.match_intent(message)
        if intent_response:
            return intent_response

        context = cls.prepare_beca_ayuda_context()
        bert_response = cls.get_bert_response(context, message)

        cls.conversation_history.append({"role": "assistant", "content": bert_response})
        return bert_response

    @classmethod
    def prepare_beca_ayuda_context(cls):
        """Prepara el contexto de becas y ayudas económicas."""
        cursor = cls.connection.cursor(dictionary=True)
        query = """
        SELECT b.nombre, b.descripcion FROM becas b
        UNION
        SELECT a.nombre, a.descripcion FROM ayudas_economicas a
        """
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

        context = "Información sobre becas y ayudas económicas en la ESPOCH: "
        for row in data:
            context += f"{row['nombre']}: {row['descripcion']}. "
        
        return context

    @classmethod
    def clear_history(cls):
        """Limpia el historial de conversación."""
        cls.conversation_history.clear()
        return "Historial de conversación borrado."

if __name__ == "__main__":
    chatbot = ChatbotService()
    print("Chatbot: Hola, soy PochiBot. ¿En qué puedo ayudarte hoy?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós', 'chao']:
            print("Chatbot: ¡Hasta luego! Espero haber sido de ayuda.")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")