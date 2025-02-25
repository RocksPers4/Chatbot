import os
import random
import logging
import mysql.connector
from mysql.connector import Error
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from config import Config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class ChatbotService:
    connection = None
    tokenizer = None
    model = None
    qa_pipeline = None
    vectorizer = None
    conversation_history = []
    stop_words = set(stopwords.words('spanish'))
    response_cache = {}

    @classmethod
    def initialize(cls):
        """Inicializa la conexión a la base de datos y el modelo."""
        cls._initialize_database()
        cls._initialize_model()

    @classmethod
    def _initialize_database(cls):
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
            cls.vectorizer.fit(cls._get_all_intents())
        except mysql.connector.Error as e:
            logging.error(f"Error al conectar a MySQL: {e}")
            raise

    @classmethod
    def _initialize_model(cls):
        logging.info("Cargando modelo DistilBERT")
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            cls.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")
            cls.qa_pipeline = pipeline("question-answering", model=cls.model, tokenizer=cls.tokenizer)
            logging.info("Modelo DistilBERT cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            cls.qa_pipeline = None
            raise

    @classmethod
    def _get_all_intents(cls):
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
    def get_bert_response(cls, context, question):
        """Genera una respuesta con DistilBERT."""
        try:
            if cls.qa_pipeline is None:
                cls._initialize_model()

            with torch.no_grad():
                max_length = 512
                context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length-50)]
                
                best_answer = ""
                best_score = 0

                for chunk in context_chunks:
                    result = cls.qa_pipeline(question=question, context=chunk, max_length=50, max_answer_length=30)

                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']

                if best_score < 0.3:
                    return ("Lo siento, no tengo suficiente información para responder a esa pregunta específica. "
                            "¿Podrías reformularla o preguntar sobre algo más general relacionado con la ESPOCH, becas o ayudas económicas? "
                            "Te recomiendo utilizar el botón de Sugerencias.")

                return best_answer.strip()

        except Exception as e:
            logging.error(f"Error al generar respuesta con DistilBERT: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu pregunta. Por favor, intenta de nuevo más tarde."

    @classmethod
    def get_response(cls, message):
        """Genera la respuesta al mensaje del usuario."""
        logging.info(f"Recibido mensaje: {message}")
        try:
            if cls.connection is None or not cls.connection.is_connected():
                cls.initialize()

            cls.conversation_history.append({"role": "user", "content": message})
            if len(cls.conversation_history) > 5:
                cls.conversation_history.pop(0)

            intent_response = cls.match_intent(message)
            if intent_response:
                return intent_response

            if message in cls.response_cache:
                logging.info("Respuesta encontrada en caché")
                return cls.response_cache[message]

            context = cls._prepare_beca_ayuda_context()
            bert_response = cls.get_bert_response(context, message)

            cls.conversation_history.append({"role": "assistant", "content": bert_response})
            cls.response_cache[message] = bert_response
            return bert_response
        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")    
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."

    @classmethod
    def _prepare_beca_ayuda_context(cls):
        """Prepara el contexto de becas y ayudas económicas."""
        try:
            cursor = cls.connection.cursor(dictionary=True)
            query = """
            SELECT 'Beca' as tipo, b.nombre, b.descripcion FROM becas b
            UNION ALL
            SELECT 'Ayuda Económica' as tipo, a.nombre, a.descripcion FROM ayudas_economicas a
            """
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()

            context = "Información sobre becas y ayudas económicas en la ESPOCH:\n"
            for row in data:
                context += f"{row['tipo']} - {row['nombre']}: {row['descripcion']}\n"
            
            context += "\nLa ESPOCH (Escuela Superior Politécnica de Chimborazo) es una institución de educación superior pública ubicada en Riobamba, Ecuador. "
            context += "Fundada en 1972, la ESPOCH se destaca por su excelencia académica y su compromiso con la investigación y el desarrollo tecnológico. "
            context += "Ofrece una amplia gama de programas de grado y posgrado en áreas como ingeniería, ciencias, administración y tecnología."

            return context
        except Exception as e:
            logging.error(f"Error al preparar el contexto: {str(e)}")
            return "Error al obtener información sobre becas y ayudas económicas."

    @classmethod
    def clear_history(cls):
        """Limpia el historial de conversación."""
        cls.conversation_history.clear()
        return "Historial de conversación borrado."

if __name__ == "__main__":
    chatbot = ChatbotService()
    chatbot.initialize()
    print("Chatbot: Hola, soy PochiBot. ¿En qué puedo ayudarte hoy?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós', 'chao']:
            print("Chatbot: ¡Hasta luego! Espero haber sido de ayuda.")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")