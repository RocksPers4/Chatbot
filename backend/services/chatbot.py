import os
import random
import logging
import mysql.connector
from mysql.connector import Error
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from config import Config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

class ChatbotService:
    connection = None
    tokenizer = None
    model = None
    qa_pipeline = None
    generation_pipeline = None
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
            
            # Cargar modelos de IA
            cls._load_models()

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
    def _load_models(cls):
        """Carga los modelos de IA."""
        try:
            # Cargar modelo de Question Answering como principal
            logging.info("Cargando modelo de Question Answering...")
            qa_model_name = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            cls.qa_pipeline = pipeline(
                "question-answering",
                model=qa_model_name,
                tokenizer=qa_model_name,
                device=-1  # CPU
            )
            logging.info("Modelo de Question Answering cargado correctamente")
            
            # Cargar modelo de generación solo como respaldo
            try:
                logging.info("Cargando modelo de generación de texto...")
                generation_model_name = "PlanTL-GOB-ES/gpt2-base-bne"
                cls.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
                cls.model = AutoModelForCausalLM.from_pretrained(generation_model_name)
                
                if cls.tokenizer.pad_token is None:
                    cls.tokenizer.pad_token = cls.tokenizer.eos_token
                
                logging.info("Modelo de generación cargado correctamente")
            except Exception as gen_error:
                logging.warning(f"No se pudo cargar el modelo de generación: {gen_error}")
                cls.model = None
                cls.tokenizer = None
            
        except Exception as e:
            logging.error(f"Error al cargar modelos avanzados: {str(e)}")
            # Fallback al modelo original
            logging.info("Cargando modelo TinyBERT como respaldo...")
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
                cls.model = AutoModelForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny")
                cls.qa_pipeline = pipeline("question-answering", model=cls.model, tokenizer=cls.tokenizer)
                logging.info("Modelo TinyBERT cargado como respaldo")
            except Exception as fallback_error:
                logging.error(f"Error al cargar modelo de respaldo: {str(fallback_error)}")
                cls.qa_pipeline = None

    @classmethod
    def _is_espoch_related(cls, message):
        """Determina si la pregunta está relacionada con ESPOCH, becas o ayudas."""
        message_lower = message.lower()
        
        # Palabras clave relacionadas con ESPOCH y becas
        espoch_keywords = [
            'beca', 'becas', 'ayuda económica', 'ayudas económicas', 'financiamiento',
            'espoch', 'universidad', 'carrera', 'carreras', 'estudios', 'estudiante',
            'matrícula', 'requisitos', 'solicitud', 'aplicar', 'postular',
            'orellana', 'sede', 'campus', 'biblioteca', 'laboratorio',
            'ingeniería', 'administración', 'enfermería', 'agronomía', 'turismo',
            'biotecnología', 'sistemas', 'computación', 'bienestar estudiantil',
            'servicios', 'académico', 'semestre', 'período académico'
        ]
        
        # Verificar si contiene palabras clave
        for keyword in espoch_keywords:
            if keyword in message_lower:
                return True
        
        return False

    @classmethod
    def get_response(cls, message):
        """Genera la respuesta al mensaje del usuario."""
        logging.info(f"Recibido mensaje: {message}")
        try:
            if cls.connection is None or not cls.connection.is_connected():
                cls.initialize()

            # Actualizar historial de conversación
            cls.conversation_history.append({"role": "user", "content": message})
            if len(cls.conversation_history) > 10:
                cls.conversation_history = cls.conversation_history[-10:]

            # 1. Buscar intent directo en la base de datos (respuestas predefinidas)
            intent_response = cls.match_intent(message)
            if intent_response:
                cls.conversation_history.append({"role": "assistant", "content": intent_response})
                return intent_response

            # 2. Verificar caché
            if message in cls.response_cache:
                logging.info("Respuesta encontrada en caché")
                return cls.response_cache[message]

            # 3. Verificar si la pregunta está relacionada con ESPOCH
            if cls._is_espoch_related(message):
                # Usar contexto específico de ESPOCH y modelo QA
                context = cls.prepare_beca_ayuda_context()
                qa_response = cls._get_qa_response(context, message)
                
                cls.conversation_history.append({"role": "assistant", "content": qa_response})
                cls.response_cache[message] = qa_response
                return qa_response
            else:
                # Para preguntas no relacionadas, dar respuesta educada pero dirigida
                general_response = cls._get_general_response(message)
                cls.conversation_history.append({"role": "assistant", "content": general_response})
                cls.response_cache[message] = general_response
                return general_response
            
        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")    
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."

    @classmethod
    def _get_general_response(cls, message):
        """Genera respuesta para preguntas no relacionadas con ESPOCH."""
        message_lower = message.lower()
        
        # Respuestas para saludos generales
        if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos']):
            return "¡Hola! Soy PochiBot, tu asistente virtual de la ESPOCH Sede Orellana. Estoy aquí para ayudarte con información sobre becas, ayudas económicas, carreras y servicios universitarios. ¿En qué puedo asistirte?"
        
        # Respuestas para despedidas
        if any(word in message_lower for word in ['adiós', 'chao', 'hasta luego', 'nos vemos', 'gracias']):
            return "¡Hasta luego! Fue un placer ayudarte. Si tienes más preguntas sobre la ESPOCH, becas o servicios universitarios, no dudes en contactarme. ¡Que tengas un excelente día!"
        
        # Respuestas para preguntas sobre el chatbot
        if any(word in message_lower for word in ['quién eres', 'qué eres', 'cómo te llamas', 'tu nombre']):
            return "Soy PochiBot, el asistente virtual de la ESPOCH Sede Orellana. Mi función es ayudar a los estudiantes con información sobre becas, ayudas económicas, carreras disponibles y servicios universitarios. ¿Hay algo específico sobre la ESPOCH en lo que pueda ayudarte?"
        
        # Para otras preguntas generales, redirigir amablemente
        return ("Gracias por tu pregunta. Soy PochiBot, especializado en ayudar con temas relacionados a la ESPOCH Sede Orellana. "
                "Puedo asistirte con información sobre:\n"
                "• Becas y ayudas económicas\n"
                "• Carreras disponibles\n"
                "• Requisitos y procesos de solicitud\n"
                "• Servicios universitarios\n"
                "• Información general de la ESPOCH\n\n"
                "¿Hay algo específico sobre estos temas en lo que pueda ayudarte?")

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

            context = "Información sobre becas y ayudas económicas en la ESPOCH Sede Orellana:\n"
            for row in data:
                context += f"{row['tipo']} - {row['nombre']}: {row['descripcion']}\n"
            
            # Información específica de ESPOCH Sede Orellana
            context += "\nLa ESPOCH Sede Orellana es una extensión de la Escuela Superior Politécnica de Chimborazo ubicada en la provincia de Orellana, Ecuador. "
            context += "Ofrece carreras como Ingeniería en Sistemas y Computación, Ingeniería en Biotecnología Ambiental, Administración de Empresas, Enfermería, Agronomía y Turismo. "
            context += "La sede cuenta con biblioteca, laboratorios especializados, departamento de bienestar estudiantil, servicios médicos y áreas deportivas. "
            context += "El período académico se divide en dos semestres: septiembre-febrero y marzo-agosto. "
            context += "Para más información, los estudiantes pueden contactar al Departamento de Bienestar Estudiantil en el edificio administrativo."

            return context
        except Exception as e:
            logging.error(f"Error al preparar el contexto: {str(e)}")
            return "Error al obtener información sobre becas y ayudas económicas."

    @classmethod
    def _get_qa_response(cls, context, question):
        """Genera una respuesta usando el modelo de Question Answering."""
        try:
            if cls.qa_pipeline is None:
                cls._load_models()
            
            with torch.no_grad():
                max_length = 512
                context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
                
                best_answer = ""
                best_score = 0

                # Procesar todos los fragmentos del contexto
                for chunk in context_chunks:
                    result = cls.qa_pipeline(question=question, context=chunk, max_length=100, max_answer_length=80)

                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']

                # Si la puntuación es alta, usar la respuesta del QA
                if best_score > 0.6:
                    return best_answer.strip()
                
                # Si la puntuación es media, dar respuesta más específica
                elif best_score > 0.3:
                    return f"{best_answer.strip()}. Para información más detallada, te recomiendo contactar al Departamento de Bienestar Estudiantil de la ESPOCH Sede Orellana."
                
                # Si la puntuación es baja, respuesta específica
                else:
                    return ("No tengo información específica sobre esa consulta en mi base de datos actual. "
                           "Te recomiendo:\n"
                           "• Contactar al Departamento de Bienestar Estudiantil\n"
                           "• Visitar la página web oficial de la ESPOCH\n"
                           "• Usar el botón de Sugerencias para ver temas disponibles\n"
                           "• Reformular tu pregunta con términos más específicos sobre becas o servicios universitarios")

        except Exception as e:
            logging.error(f"Error al generar respuesta con QA: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu pregunta. Por favor, intenta de nuevo más tarde."

    @classmethod
    def handle_feedback(cls, feedback, last_response):
        """Maneja el feedback del usuario."""
        try:
            if feedback.lower() in ['bueno', 'útil', 'correcto', 'gracias', 'excelente']:
                return "¡Gracias por tu feedback positivo! Me alegra haber sido de ayuda. ¿Hay algo más sobre la ESPOCH en lo que pueda asistirte?"
            elif feedback.lower() in ['malo', 'incorrecto', 'no útil', 'error']:
                return "Lamento que la respuesta no haya sido útil. Estoy aprendiendo constantemente para mejorar. ¿Podrías reformular tu pregunta o ser más específico sobre lo que necesitas saber?"
            else:
                return "Gracias por tu feedback. Lo tomaré en cuenta para mejorar mis respuestas. ¿Hay algo más en lo que pueda ayudarte?"
        except Exception as e:
            logging.error(f"Error al procesar feedback: {str(e)}")
            return "Gracias por tu feedback."

    @classmethod
    def clear_history(cls):
        """Limpia el historial de conversación."""
        cls.conversation_history.clear()
        cls.response_cache.clear()
        return "Historial de conversación borrado."

if __name__ == "__main__":
    ChatbotService.initialize()
    print("Chatbot: Hola, soy PochiBot. ¿En qué puedo ayudarte hoy?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós', 'chao']:
            print("Chatbot: ¡Hasta luego! Espero haber sido de ayuda.")
            break
        response = ChatbotService.get_response(user_input)
        print(f"Chatbot: {response}")
