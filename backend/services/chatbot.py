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
            # Cargar modelo de generación de texto en español
            logging.info("Cargando modelo de generación de texto...")
            generation_model_name = "PlanTL-GOB-ES/gpt2-base-bne"
            cls.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(generation_model_name)
            
            # Configurar pad_token si no existe
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            
            # No usar pipeline, usar el modelo directamente para mejor control
            logging.info("Modelo de generación cargado correctamente")
            
            # Cargar modelo de Question Answering como respaldo
            logging.info("Cargando modelo de Question Answering...")
            qa_model_name = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            cls.qa_pipeline = pipeline(
                "question-answering",
                model=qa_model_name,
                tokenizer=qa_model_name,
                device=-1  # CPU
            )
            logging.info("Modelo de Question Answering cargado correctamente")
            
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

            # 3. Generar respuesta con el modelo de lenguaje
            generated_response = cls._generate_ai_response(message)
            if generated_response:
                cls.conversation_history.append({"role": "assistant", "content": generated_response})
                cls.response_cache[message] = generated_response
                return generated_response

            # 4. Si la generación falla, usar el modelo de QA con contexto
            context = cls.prepare_beca_ayuda_context()
            bert_response = cls._get_qa_response(context, message)
            
            cls.conversation_history.append({"role": "assistant", "content": bert_response})
            cls.response_cache[message] = bert_response
            return bert_response
            
        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")    
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."
        
    @classmethod
    def _generate_ai_response(cls, message):
        """Genera una respuesta usando el modelo de lenguaje."""
        try:
            if cls.model is None or cls.tokenizer is None:
                return None
                
            # Preparar el contexto con el historial de conversación (más corto)
            conversation_context = ""
            # Usar solo las últimas 2 interacciones para evitar prompts muy largos
            recent_history = cls.conversation_history[-4:] if len(cls.conversation_history) > 4 else cls.conversation_history
            
            for item in recent_history:
                role = "Usuario: " if item["role"] == "user" else "PochiBot: "
                conversation_context += role + item["content"] + "\n"
            
            # Prompt más corto y directo
            system_prompt = "PochiBot ayuda a estudiantes de ESPOCH Orellana con becas y servicios universitarios.\n\n"
            
            # Crear el prompt completo
            prompt = system_prompt + conversation_context + "PochiBot: "
            
            # Tokenizar el prompt
            inputs = cls.tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)
            
            # Generar respuesta usando el modelo directamente
            with torch.no_grad():
                outputs = cls.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.8,
                    top_p=0.92,
                    do_sample=True,
                    pad_token_id=cls.tokenizer.eos_token_id,
                    eos_token_id=cls.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decodificar la respuesta
            generated_text = cls.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraer solo la respuesta generada
            response = generated_text.split("PochiBot: ")[-1].strip()
            
            # Verificar que la respuesta sea adecuada
            if len(response) > 10 and "Usuario:" not in response and response != prompt:
                # Limpiar posibles artefactos
                response = response.split("\n")[0]
                # Limitar longitud de respuesta
                if len(response) > 200:
                    response = response[:200] + "..."
                return response
                
            return None
            
        except Exception as e:
            logging.error(f"Error al generar respuesta con IA: {str(e)}")
            return None
            
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
    def _get_qa_response(cls, context, question):
        """Genera una respuesta usando el modelo de Question Answering."""
        try:
            # Cargar los modelos si aún no se ha cargado el pipeline
            if cls.qa_pipeline is None:
                cls._load_models()
            
            # Prevenir el cálculo de gradientes
            with torch.no_grad():
                max_length = 512  # Limite máximo de tokens por fragmento
                context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
                
                best_answer = ""
                best_score = 0  # Empezar con puntaje cero

                # Procesar todos los fragmentos del contexto
                for chunk in context_chunks:
                    result = cls.qa_pipeline(question=question, context=chunk, max_length=50, max_answer_length=30)

                    # Comprobar el puntaje y comparar
                    if result['score'] > best_score:
                        best_answer = result['answer']
                        best_score = result['score']

                # Si la puntuación es muy baja, dar respuesta genérica
                if best_score < 0.5:  
                    return ("Lo siento, no tengo suficiente información para responder a esa pregunta específica. "
                            "¿Podrías reformularla o preguntar sobre algo más general relacionado con la ESPOCH, becas o ayudas económicas?."
                            "Te recomiendo utilizar el botón de Sugerencias")

                return best_answer.strip()

        except (ValueError, KeyError) as e:
            logging.error(f"Error al generar respuesta con QA: {str(e)}")
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
