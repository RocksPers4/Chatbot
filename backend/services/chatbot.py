import os
import random
import logging
import mysql.connector
from mysql.connector import Error
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class ChatbotService:
    model_name = "nikravan/glm-4vq"
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
        """Carga el modelo GLM-4VQ solo si es necesario."""
        if cls.model is None or cls.tokenizer is None:
            logging.info("Cargando modelo GLM-4VQ")
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, trust_remote_code=True)
                cls.model = AutoModelForCausalLM.from_pretrained(
                    cls.model_name, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"  # Mueve automáticamente el modelo a GPU si está disponible
                )
                logging.info("Modelo GLM-4VQ cargado correctamente")
            except Exception as e:
                logging.error(f"Error al cargar el modelo: {str(e)}")
                cls.model = None
                cls.tokenizer = None
                raise

    @classmethod
    def get_glm_response(cls, context, question):
        """Genera una respuesta con GLM-4VQ."""
        try:
            if cls.model is None or cls.tokenizer is None:
                cls.load_models()

            prompt = f"Contexto: {context}\nPregunta: {question}\nRespuesta:"
            inputs = cls.tokenizer(prompt, return_tensors="pt").to(cls.model.device)

            with torch.no_grad():
                output = cls.model.generate(**inputs, max_length=100, num_return_sequences=1)

            best_answer = cls.tokenizer.decode(output[0], skip_special_tokens=True)
            best_answer = best_answer.split("Respuesta:")[-1].strip()

            if len(best_answer) < 10:
                return "Lo siento, no tengo suficiente información para responder a esa pregunta."

            return best_answer

        except Exception as e:
            logging.error(f"Error al generar respuesta con GLM-4VQ: {str(e)}")
            return "Lo siento, ha ocurrido un error inesperado. Intenta nuevamente más tarde."
        
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

            # Verificar si la respuesta está en caché
            if message in cls.response_cache:
                logging.info("Respuesta encontrada en caché")
                return cls.response_cache[message]

            context = cls.prepare_beca_ayuda_context()
            bert_response = cls.get_bert_response(context, message)

            cls.conversation_history.append({"role": "assistant", "content": bert_response})
            cls.response_cache[message] = bert_response  # Corregido: usar bert_response en lugar de response
            return bert_response  # Añadido: retornar bert_response
        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")    
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."

    @classmethod
    def prepare_beca_ayuda_context(cls):
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
            
            # Añadir información general sobre la ESPOCH
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
    print("Chatbot: Hola, soy PochiBot. ¿En qué puedo ayudarte hoy?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós', 'chao']:
            print("Chatbot: ¡Hasta luego! Espero haber sido de ayuda.")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")