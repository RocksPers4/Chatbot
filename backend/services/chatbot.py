import logging
import mysql.connector
import torch
import requests
import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
from config import Config
from services.knowledge_manager import knowledge_manager

logging.basicConfig(level=logging.INFO)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ChatbotService:
    connection = None
    tokenizer = None
    model = None
    qa_pipeline = None
    conversation_history = []
    response_cache = {}
    last_question = None
    last_response = None
    
    # Configuración DeepSeek-R1
    deepseek_api_key = os.getenv('sk-a024882a24cd4faf9c2743ceffe42926')
    deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
    
    # Respuestas rápidas predefinidas
    quick_responses = {
        "qué es la espoch": """La ESPOCH (Escuela Superior Politécnica de Chimborazo) es una institución pública de educación superior fundada en 1969.

• Una de las principales universidades técnicas de Ecuador
• Campus principal en Riobamba, Sede Orellana en la Amazonía
• Acreditada por el CACES
• Enfoque en carreras técnicas e ingenierías

La Sede Orellana ofrece educación superior de calidad en la región amazónica.""",
        
        "carreras espoch orellana": """La ESPOCH Sede Orellana ofrece 5 carreras:

• Agronomía
• Turismo  
• Ingeniería Ambiental
• Zootecnia
• Tecnologías de la Información

Todas con infraestructura moderna y convenios para prácticas."""
    }

    @classmethod
    def initialize(cls):
        """Inicialización con DeepSeek-R1."""
        try:
            cls.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                port=Config.MYSQL_PORT
            )
            
            # Verificar si DeepSeek está disponible
            if cls.deepseek_api_key:
                cls._test_deepseek_connection()
                logging.info("DeepSeek-R1 configurado correctamente")
            else:
                logging.warning("DEEPSEEK_API_KEY no encontrada, usando modelos locales")
                cls._load_fallback_models()
            
            logging.info("ChatbotService inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando ChatbotService: {e}")
            raise

    @classmethod
    def _test_deepseek_connection(cls):
        """Prueba la conexión con DeepSeek API."""
        try:
            headers = {
                "Authorization": f"Bearer {cls.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            
            response = requests.post(cls.deepseek_api_url, 
                                   headers=headers, 
                                   json=test_payload, 
                                   timeout=5)
            
            if response.status_code == 200:
                logging.info("Conexión con DeepSeek-R1 exitosa")
                return True
            else:
                logging.warning(f"Error en DeepSeek API: {response.status_code}")
                cls._load_fallback_models()
                return False
                
        except Exception as e:
            logging.warning(f"No se pudo conectar con DeepSeek: {e}")
            cls._load_fallback_models()
            return False

    @classmethod
    def _load_fallback_models(cls):
        """Carga modelos locales como respaldo."""
        try:
            # Modelo GODEL como respaldo
            model_name = "microsoft/GODEL-v1_1-base-seq2seq"
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Modelo QA
            qa_model = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            cls.qa_pipeline = pipeline("question-answering", model=qa_model, device=-1)
            
            logging.info("Modelos de respaldo cargados correctamente")
        except Exception as e:
            logging.error(f"Error cargando modelos de respaldo: {e}")
            cls.qa_pipeline = None

    @classmethod
    def get_response(cls, message):
        """Generador de respuestas con DeepSeek-R1."""
        logging.info(f"Mensaje recibido: {message}")
        
        try:
            cls.last_question = message
            cls._update_history("user", message)
            
            # 1. Conocimiento aprendido (prioridad alta)
            knowledge_answer = knowledge_manager.get_answer(message, threshold=0.75)
            if knowledge_answer and knowledge_answer['confianza'] > 0.6:
                response = knowledge_answer['respuesta']
                cls.last_response = response
                cls._update_history("assistant", response)
                return response

            # 2. Respuestas rápidas predefinidas
            quick_response = cls._check_quick_responses(message)
            if quick_response:
                cls.last_response = quick_response
                cls._update_history("assistant", quick_response)
                return quick_response

            # 3. Base de datos tradicional
            db_response = cls._get_db_response(message)
            if db_response:
                cls.last_response = db_response
                cls._update_history("assistant", db_response)
                return db_response

            # 4. DeepSeek-R1 para respuestas complejas
            if cls._is_espoch_related(message):
                if cls.deepseek_api_key:
                    deepseek_response = cls._get_deepseek_response(message)
                    if deepseek_response:
                        cls.last_response = deepseek_response
                        cls._update_history("assistant", deepseek_response)
                        return deepseek_response
                
                # Fallback a modelo local
                ai_response = cls._get_local_ai_response(message)
            else:
                ai_response = cls._get_general_response(message)
            
            cls.last_response = ai_response
            cls._update_history("assistant", ai_response)
            return ai_response
            
        except Exception as e:
            logging.error(f"Error generando respuesta: {e}")
            error_msg = "Lo siento, ha ocurrido un error. Por favor, intenta de nuevo."
            cls.last_response = error_msg
            return error_msg

    @classmethod
    def _get_deepseek_response(cls, message):
        """Genera respuesta usando DeepSeek-R1."""
        try:
            headers = {
                "Authorization": f"Bearer {cls.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            # Construir contexto específico para ESPOCH
            context = cls._get_espoch_context()
            
            # Construir historial de conversación
            messages = [
                {
                    "role": "system",
                    "content": f"""Eres PochiBot, el asistente virtual oficial de la ESPOCH Sede Orellana. 

CONTEXTO INSTITUCIONAL:
{context}

INSTRUCCIONES:
- Responde de manera amigable y profesional
- Usa información específica de la ESPOCH Sede Orellana
- Estructura tus respuestas con viñetas (•) cuando sea apropiado
- Si no tienes información específica, recomienda contactar Bienestar Estudiantil
- Mantén respuestas concisas pero informativas (máximo 200 palabras)
- Enfócate en becas, ayudas económicas, carreras y servicios universitarios"""
                }
            ]
            
            # Añadir historial reciente
            for item in cls.conversation_history[-6:]:  # Últimas 3 interacciones
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
            
            # Añadir mensaje actual
            messages.append({
                "role": "user",
                "content": message
            })
            
            payload = {
                "model": "deepseek-reasoner",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(cls.deepseek_api_url, 
                                   headers=headers, 
                                   json=payload, 
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extraer respuesta y razonamiento si está disponible
                if 'choices' in result and len(result['choices']) > 0:
                    choice = result['choices'][0]
                    content = choice['message']['content']
                    
                    # Si hay razonamiento disponible, lo registramos pero no lo mostramos al usuario
                    if 'reasoning' in choice['message']:
                        reasoning = choice['message']['reasoning']
                        logging.info(f"DeepSeek reasoning: {reasoning[:100]}...")
                    
                    # Limpiar y formatear respuesta
                    formatted_response = cls._format_deepseek_response(content)
                    
                    logging.info("Respuesta generada con DeepSeek-R1")
                    return formatted_response
                else:
                    logging.warning("Respuesta de DeepSeek sin contenido válido")
                    return None
            else:
                logging.error(f"Error en DeepSeek API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error("Timeout en DeepSeek API")
            return None
        except Exception as e:
            logging.error(f"Error en DeepSeek-R1: {e}")
            return None

    @classmethod
    def _format_deepseek_response(cls, content):
        """Formatea la respuesta de DeepSeek para mejor presentación."""
        try:
            # Limpiar respuesta
            content = content.strip()
            
            # Asegurar que termine con pregunta de seguimiento si es muy corta
            if len(content) < 100 and not content.endswith('?'):
                content += "\n\n¿Hay algo más específico en lo que pueda ayudarte?"
            
            # Mejorar formato de listas si es necesario
            if '\n-' in content:
                content = content.replace('\n-', '\n•')
            
            return content
            
        except Exception as e:
            logging.error(f"Error formateando respuesta DeepSeek: {e}")
            return content

    @classmethod
    def _get_espoch_context(cls):
        """Contexto específico de ESPOCH para DeepSeek."""
        return """ESPOCH SEDE ORELLANA - INFORMACIÓN INSTITUCIONAL:

SOBRE LA INSTITUCIÓN:
• Escuela Superior Politécnica de Chimborazo - Sede Orellana
• Universidad pública ubicada en la provincia de Orellana, Ecuador
• Extensión de la ESPOCH principal (Riobamba)
• Enfoque en educación superior técnica y científica en la región amazónica

CARRERAS DISPONIBLES:
• Agronomía
• Turismo
• Ingeniería Ambiental  
• Zootecnia
• Tecnologías de la Información

SERVICIOS ESTUDIANTILES:
• Becas de excelencia académica
• Becas socioeconómicas
• Ayudas económicas (alimentación, transporte, materiales)
• Biblioteca especializada
• Laboratorios de computación y ciencias
• Departamento de Bienestar Estudiantil
• Servicios médicos básicos
• Áreas deportivas y recreativas

PERÍODOS ACADÉMICOS:
• Primer semestre: Septiembre - Febrero
• Segundo semestre: Marzo - Agosto
• Convocatorias de becas: Marzo-Abril y Agosto-Septiembre

REQUISITOS GENERALES PARA BECAS:
• Promedio mínimo 8.5 puntos
• Sin materias perdidas o reprobadas
• Certificado de notas actualizado
• Documentación socioeconómica (para becas socioeconómicas)
• Carta de motivación

CONTACTO:
• Departamento de Bienestar Estudiantil
• Edificio administrativo de la ESPOCH Sede Orellana"""

    @classmethod
    def _check_quick_responses(cls, message):
        """Verifica respuestas rápidas."""
        message_lower = message.lower()
        for key, response in cls.quick_responses.items():
            if key in message_lower:
                return response
        return None

    @classmethod
    def _get_db_response(cls, message):
        """Busca en base de datos tradicional."""
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
            ) AS respuestas LIMIT 1
            """
            cursor.execute(query, (f"%{message.lower()}%", f"%{message.lower()}%"))
            result = cursor.fetchone()
            cursor.close()
            return result["respuesta"] if result else None
        except Exception as e:
            logging.error(f"Error en BD: {e}")
            return None

    @classmethod
    def _is_espoch_related(cls, message):
        """Verifica si es relacionado con ESPOCH."""
        keywords = ['beca', 'becas', 'ayuda', 'espoch', 'universidad', 'carrera', 
                   'matrícula', 'orellana', 'sede', 'estudiante', 'académico',
                   'agronomía', 'turismo', 'ingeniería', 'zootecnia', 'tecnologías']
        return any(keyword in message.lower() for keyword in keywords)

    @classmethod
    def _get_local_ai_response(cls, message):
        """Genera respuesta con modelo local (fallback)."""
        try:
            if not cls.model or not cls.tokenizer:
                return cls._get_fallback_response(message)
            
            context = cls._get_espoch_context()
            prompt = f"Eres PochiBot de ESPOCH Sede Orellana. Contexto: {context} Pregunta: {message} Respuesta:"
            
            inputs = cls.tokenizer.encode_plus(prompt, return_tensors='pt', 
                                             truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = cls.model.generate(inputs['input_ids'], 
                                           max_new_tokens=100, 
                                           temperature=0.7, 
                                           do_sample=True)
            
            response = cls.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Respuesta:")[-1].strip()
            
            return response if len(response) > 20 else cls._get_fallback_response(message)
            
        except Exception as e:
            logging.error(f"Error en IA local: {e}")
            return cls._get_fallback_response(message)

    @classmethod
    def _get_general_response(cls, message):
        """Respuestas para temas no relacionados."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['hola', 'buenos', 'saludos']):
            return ("¡Hola! Soy PochiBot de la ESPOCH Sede Orellana.\n\n"
                   "Te ayudo con:\n• Becas y ayudas económicas\n• Información de carreras\n"
                   "• Trámites académicos\n• Servicios universitarios\n\n¿En qué puedo ayudarte?")
        
        if any(word in message_lower for word in ['adiós', 'chao', 'gracias']):
            return ("¡Hasta luego! Fue un placer ayudarte.\n\n"
                   "Recuerda que estoy disponible para consultas sobre la ESPOCH. ¡Que tengas un excelente día!")
        
        return ("Soy PochiBot, especializado en temas de la ESPOCH Sede Orellana.\n\n"
               "Puedo ayudarte con:\n• Becas y ayudas económicas\n• Carreras disponibles\n"
               "• Trámites y requisitos\n• Servicios universitarios\n\n¿En qué área necesitas información?")

    @classmethod
    def _get_fallback_response(cls, message):
        """Respuesta de respaldo."""
        if cls._is_espoch_related(message):
            return ("Te recomiendo:\n• Contactar Bienestar Estudiantil\n"
                   "• Visitar la página web oficial\n• Reformular tu pregunta\n"
                   "¿Hay algo específico sobre becas o carreras en lo que pueda ayudarte?")
        return cls._get_general_response(message)

    @classmethod
    def _update_history(cls, role, content):
        """Actualiza historial de conversación."""
        cls.conversation_history.append({"role": role, "content": content})
        if len(cls.conversation_history) > 12:  # Más historial para DeepSeek
            cls.conversation_history = cls.conversation_history[-12:]

    @classmethod
    def handle_feedback(cls, feedback, last_response):
        """Maneja feedback con autoaprendizaje."""
        try:
            if cls.last_question and cls.last_response:
                knowledge_manager.learn_from_conversation(
                    cls.last_question, cls.last_response, feedback)
            
            if feedback.lower() in ['bueno', 'útil', 'correcto', 'gracias', 'excelente']:
                return "¡Gracias por tu feedback! Me alegra haber sido útil. ¿Hay algo más en lo que pueda ayudarte?"
            else:
                return "Gracias por tu feedback. Trabajaré para mejorar mis respuestas. ¿Puedes reformular tu pregunta de manera más específica?"
        except Exception as e:
            logging.error(f"Error en feedback: {e}")
            return "Gracias por tu feedback."

    @classmethod
    def clear_history(cls):
        """Limpia historial."""
        cls.conversation_history.clear()
        cls.response_cache.clear()
        cls.last_question = None
        cls.last_response = None
        return "Historial borrado."

    @classmethod
    def get_model_status(cls):
        """Obtiene el estado de los modelos."""
        status = {
            "deepseek_available": bool(cls.deepseek_api_key),
            "local_models_loaded": bool(cls.model and cls.tokenizer),
            "qa_pipeline_loaded": bool(cls.qa_pipeline),
            "database_connected": bool(cls.connection and cls.connection.is_connected())
        }
        return status

if __name__ == "__main__":
    ChatbotService.initialize()
    print("PochiBot con DeepSeek-R1: ¡Hola! ¿En qué puedo ayudarte?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós']:
            print("PochiBot: ¡Hasta luego!")
            break
        response = ChatbotService.get_response(user_input)
        print(f"PochiBot: {response}")
