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
torch.set_num_threads(2)  # Aumentado para mejor rendimiento
torch.set_num_interop_threads(1)

class ChatbotService:
    connection = None
    tokenizer = None
    model = None
    qa_pipeline = None
    generation_pipeline = None  # Nuevo: pipeline de generación
    vectorizer = None
    conversation_history = []
    stop_words = set(stopwords.words('spanish'))
    response_cache = {}
    
    # Nueva: Base de conocimientos expandida para preguntas generales
    knowledge_base = {
        "general_espoch": [
            "La ESPOCH Sede Orellana está ubicada en la provincia de Orellana, Ecuador, en la región amazónica.",
            "Ofrece carreras en áreas de ingeniería, administración, ciencias de la salud y tecnología.",
            "El período académico está dividido en dos semestres por año: septiembre-febrero y marzo-agosto.",
            "La biblioteca está abierta de lunes a viernes de 7:00 a 21:00 y sábados de 8:00 a 16:00.",
            "El campus cuenta con laboratorios especializados, áreas deportivas, cafetería y residencia estudiantil."
        ],
        "carreras": [
            "Ingeniería en Tecnologías de la Información",
            "Ingeniería en Biotecnología Ambiental", 
            "Ingeniería en Zootecnia",
            "Ingeniería Ambiental",
            "Agronomía",
            "Turismo"
        ],
        "servicios": [
            "Biblioteca con acceso a bases de datos académicas",
            "Laboratorios de computación y ciencias",
            "Servicio médico estudiantil",
            "Cafetería y comedor estudiantil",
            "Áreas deportivas y recreativas",
            "Departamento de Bienestar Estudiantil"
        ]
    }

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
        """Carga los modelos de IA mejorados."""
        if cls.qa_pipeline is None:
            try:
                # Cargar modelo de Question Answering en español (mejorado)
                logging.info("Cargando modelo de Question Answering en español...")
                qa_model_name = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
                cls.qa_pipeline = pipeline(
                    "question-answering",
                    model=qa_model_name,
                    tokenizer=qa_model_name,
                    device=-1  # CPU
                )
                logging.info("Modelo de Question Answering cargado correctamente")
                
                # Cargar modelo de generación para respuestas más naturales
                logging.info("Cargando modelo de generación de texto...")
                generation_model_name = "PlanTL-GOB-ES/gpt2-base-bne"
                cls.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
                cls.model = AutoModelForCausalLM.from_pretrained(generation_model_name)
                
                # Configurar pad_token si no existe
                if cls.tokenizer.pad_token is None:
                    cls.tokenizer.pad_token = cls.tokenizer.eos_token
                
                cls.generation_pipeline = pipeline(
                    "text-generation",
                    model=cls.model,
                    tokenizer=cls.tokenizer,
                    device=-1,
                    max_length=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=cls.tokenizer.eos_token_id
                )
                logging.info("Modelo de generación cargado correctamente")
                
            except Exception as e:
                logging.error(f"Error al cargar modelos mejorados: {str(e)}")
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
    def _detect_general_intent(cls, message):
        """Detecta si la pregunta es sobre temas generales de la ESPOCH."""
        message_lower = message.lower()
        
        general_keywords = {
            "carreras": ["carrera", "carreras", "estudiar", "programa", "ingeniería", "administración", "ambiental"],
            "servicios": ["servicio", "servicios", "biblioteca", "laboratorio", "cafetería", "comedor", "deporte"],
            "general_espoch": ["espoch", "universidad", "campus", "sede", "orellana", "ubicación", "información"]
        }
        
        for category, keywords in general_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return category
        return None

    @classmethod
    def _get_knowledge_response(cls, category, message):
        """Obtiene una respuesta de la base de conocimientos."""
        if category in cls.knowledge_base:
            responses = cls.knowledge_base[category]
            # Seleccionar respuesta más relevante usando similitud
            try:
                if cls.vectorizer:
                    message_vec = cls.vectorizer.transform([message])
                    response_vecs = cls.vectorizer.transform(responses)
                    similarities = cosine_similarity(message_vec, response_vecs)[0]
                    best_idx = np.argmax(similarities)
                    return responses[best_idx]
            except:
                pass
            return random.choice(responses)
        return None

    @classmethod
    def get_response(cls, message):
        """Genera la respuesta al mensaje del usuario."""
        logging.info(f"Recibido mensaje: {message}")
        try:
            if cls.connection is None or not cls.connection.is_connected():
                cls.initialize()

            cls.conversation_history.append({"role": "user", "content": message})

            if len(cls.conversation_history) > 10:  # Aumentado el historial
                cls.conversation_history = cls.conversation_history[-10:]

            # 1. Buscar intent directo en la base de datos
            intent_response = cls.match_intent(message)
            if intent_response:
                return intent_response

            # 2. Verificar caché
            if message in cls.response_cache:
                logging.info("Respuesta encontrada en caché")
                return cls.response_cache[message]

            # 3. Detectar si es pregunta general
            general_category = cls._detect_general_intent(message)
            if general_category:
                knowledge_response = cls._get_knowledge_response(general_category, message)
                if knowledge_response:
                    cls.response_cache[message] = knowledge_response
                    return knowledge_response

            # 4. Usar contexto de becas y IA
            context = cls.prepare_beca_ayuda_context()
            bert_response = cls.get_bert_response(context, message)

            cls.conversation_history.append({"role": "assistant", "content": bert_response})
            cls.response_cache[message] = bert_response
            return bert_response
            
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

            context = "Información sobre becas y ayudas económicas en la ESPOCH Sede Orellana:\n"
            for row in data:
                context += f"{row['tipo']} - {row['nombre']}: {row['descripcion']}\n"
            
            # Información expandida sobre la ESPOCH
            context += "\nLa ESPOCH Sede Orellana es una extensión de la Escuela Superior Politécnica de Chimborazo ubicada en la provincia de Orellana, Ecuador. "
            context += "Se especializa en carreras relacionadas con la región amazónica como Ingeniería en Biotecnología Ambiental, Agronomía, y otras. "
            context += "Ofrece servicios estudiantiles completos incluyendo biblioteca, laboratorios, bienestar estudiantil y actividades deportivas."

            return context
        except Exception as e:
            logging.error(f"Error al preparar el contexto: {str(e)}")
            return "Error al obtener información sobre becas y ayudas económicas."

    @classmethod
    def get_bert_response(cls, context, question):
        """Genera una respuesta mejorada con los nuevos modelos."""
        try:
            # Cargar los modelos si aún no se ha cargado el pipeline
            if cls.qa_pipeline is None:
                cls.load_models()
            
            # Prevenir el cálculo de gradientes
            with torch.no_grad():
                max_length = 512
                context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
                
                best_answer = ""
                best_score = 0

                # Procesar con el modelo de QA mejorado
                for chunk in context_chunks:
                    result = cls.qa_pipeline(question=question, context=chunk, max_length=100, max_answer_length=50)

                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']

                # Si la puntuación es alta, usar la respuesta del QA
                if best_score > 0.6:
                    return best_answer.strip()
                
                # Si la puntuación es media, intentar generar respuesta más natural
                elif best_score > 0.3 and cls.generation_pipeline:
                    try:
                        prompt = f"Pregunta: {question}\nContexto: {context[:200]}...\nRespuesta:"
                        generated = cls.generation_pipeline(
                            prompt,
                            max_length=len(prompt.split()) + 30,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=cls.tokenizer.eos_token_id
                        )
                        
                        response = generated[0]['generated_text'].split("Respuesta:")[-1].strip()
                        if len(response) > 10 and len(response) < 200:
                            return response
                    except Exception as gen_error:
                        logging.error(f"Error en generación: {gen_error}")
                
                # Respuesta genérica mejorada
                return ("Lo siento, no tengo información específica sobre esa consulta. "
                       "Puedo ayudarte con información sobre becas, ayudas económicas, carreras disponibles, "
                       "servicios universitarios y aspectos generales de la ESPOCH Sede Orellana. "
                       "Te recomiendo utilizar el botón de Sugerencias para ver temas disponibles.")

        except (ValueError, KeyError) as e:
            logging.error(f"Error al generar respuesta: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu pregunta. Por favor, intenta de nuevo más tarde."

        except Exception as e:
            logging.error(f"Error desconocido: {str(e)}")
            return "Lo siento, ha ocurrido un error inesperado. Intenta nuevamente más tarde."

    @classmethod
    def handle_feedback(cls, feedback, last_response):
        """Maneja el feedback del usuario."""
        try:
            if feedback.lower() in ['bueno', 'útil', 'correcto', 'gracias', 'excelente']:
                return "¡Gracias por tu feedback positivo! Me alegra haber sido de ayuda."
            elif feedback.lower() in ['malo', 'incorrecto', 'no útil', 'error']:
                return "Lamento que la respuesta no haya sido útil. Estoy aprendiendo constantemente para mejorar."
            else:
                return "Gracias por tu feedback. Lo tomaré en cuenta para mejorar mis respuestas."
        except Exception as e:
            logging.error(f"Error al procesar feedback: {str(e)}")
            return "Gracias por tu feedback."

    @classmethod
    def clear_history(cls):
        """Limpia el historial de conversación."""
        cls.conversation_history.clear()
        cls.response_cache.clear()  # También limpiar caché
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
