import os
import random
import logging
import mysql.connector
from mysql.connector import Error
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering, 
    pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from config import Config
from services.knowledge_manager import knowledge_manager

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
    vectorizer = None
    conversation_history = []
    stop_words = set(stopwords.words('spanish'))
    response_cache = {}
    
    # Variable para almacenar la última respuesta para el feedback
    last_question = None
    last_response = None
    
    # Respuestas predefinidas para preguntas frecuentes
    common_questions = {
        "qué es la espoch": """
La ESPOCH (Escuela Superior Politécnica de Chimborazo) es una institución pública de educación superior fundada el 18 de abril de 1969.

• Es una de las principales universidades técnicas de Ecuador
• Su campus principal está ubicado en Riobamba, provincia de Chimborazo
• La Sede Orellana es una extensión que ofrece educación superior de calidad en la región amazónica
• Está acreditada por el CACES (Consejo de Aseguramiento de la Calidad de la Educación Superior)
• Se destaca por su enfoque en carreras técnicas, ingenierías y ciencias aplicadas

La ESPOCH Sede Orellana fue creada para ampliar la oferta académica en la Amazonía ecuatoriana, brindando acceso a educación superior de calidad en la región.
        """,
        
        "historia de la espoch": """
Historia de la ESPOCH:

• Fundada el 18 de abril de 1969 como Instituto Superior Tecnológico de Chimborazo
• En 1972 se transformó en Escuela Superior Politécnica de Chimborazo (ESPOCH)
• Inició con las Facultades de Ingeniería y Zootecnia
• A lo largo de los años ha ampliado su oferta académica y su infraestructura
• La Sede Orellana fue creada para atender las necesidades educativas de la región amazónica
• Actualmente es reconocida como una de las mejores universidades técnicas del país

La ESPOCH ha formado a miles de profesionales que contribuyen al desarrollo del Ecuador en diversos campos científicos y tecnológicos.
        """,
        
        "misión de la espoch": """
Misión de la ESPOCH:

"Formar profesionales e investigadores competentes, para contribuir al desarrollo sostenible del país."

Esta misión se sustenta en:
• Excelencia académica
• Investigación científica y tecnológica
• Vinculación con la sociedad
• Gestión administrativa eficiente
• Compromiso con el desarrollo sostenible
• Formación integral de los estudiantes

La ESPOCH Sede Orellana trabaja bajo esta misma misión, adaptándola a las necesidades específicas de la región amazónica.
        """,
        
        "visión de la espoch": """
Visión de la ESPOCH:

"Ser una institución de educación superior líder en la Zona 3 del Ecuador, con reconocimiento nacional y proyección internacional."

Esta visión busca:
• Posicionar a la ESPOCH como referente de calidad educativa
• Desarrollar investigación científica de impacto
• Formar profesionales altamente competitivos
• Contribuir al desarrollo sostenible del país
• Establecer vínculos con la sociedad y el sector productivo
• Proyectarse internacionalmente a través de convenios y colaboraciones

La Sede Orellana comparte esta visión, enfocándose en ser un referente educativo en la región amazónica.
        """,
        
        "espoch sede orellana": """
ESPOCH Sede Orellana:

• Es una extensión de la Escuela Superior Politécnica de Chimborazo
• Ubicada en la provincia de Orellana, región amazónica de Ecuador
• Creada para ampliar la oferta académica en la Amazonía ecuatoriana
• Ofrece 5 carreras de pregrado: Agronomía, Turismo, Ingeniería Ambiental, Zootecnia y Tecnologías de la Información
• Cuenta con infraestructura moderna: aulas, laboratorios, biblioteca y áreas recreativas
• Tiene convenios con instituciones locales para prácticas pre-profesionales
• Brinda servicios de bienestar estudiantil y becas

La Sede Orellana representa el compromiso de la ESPOCH con la descentralización de la educación superior de calidad.
        """
    }

    # Configuración de modelos disponibles (sin dependencias problemáticas)
    AVAILABLE_MODELS = {
        "blenderbot": {
            "name": "facebook/blenderbot-400M-distill",
            "type": "causal",
            "description": "BlenderBot: Especializado en conversaciones naturales y empáticas",
            "dependencies": []
        },
        "spanish_gpt2": {
            "name": "PlanTL-GOB-ES/gpt2-base-bne",
            "type": "causal", 
            "description": "GPT-2 Español: Modelo generativo entrenado específicamente en español",
            "dependencies": []
        },
        "distilgpt2": {
            "name": "distilgpt2",
            "type": "causal",
            "description": "DistilGPT-2: Modelo ligero y rápido para conversaciones",
            "dependencies": []
        },
        "gpt2_medium": {
            "name": "gpt2-medium",
            "type": "causal",
            "description": "GPT-2 Medium: Modelo más grande para mejores respuestas",
            "dependencies": []
        },
        "dialogpt": {
            "name": "microsoft/DialoGPT-medium",
            "type": "causal",
            "description": "DialoGPT: Especializado en diálogos conversacionales",
            "dependencies": []
        }
    }
    
    # Modelo actual seleccionado (cambiado a uno sin dependencias)
    CURRENT_MODEL = "blenderbot"  # Modelo más estable

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
        """Carga los modelos de IA según la configuración."""
        try:
            model_config = cls.AVAILABLE_MODELS[cls.CURRENT_MODEL]
            model_name = model_config["name"]
            
            logging.info(f"Cargando modelo principal: {model_name}")
            logging.info(f"Descripción: {model_config['description']}")
            
            # Cargar modelo principal (todos son causales ahora)
            cls._load_causal_model(model_name)
            
            # Cargar modelo QA en español como respaldo
            cls._load_qa_model()
            
            logging.info(f"Modelo {cls.CURRENT_MODEL} cargado correctamente")
            
        except Exception as e:
            logging.error(f"Error al cargar modelo {cls.CURRENT_MODEL}: {str(e)}")
            # Fallback a modelo más simple
            cls._load_fallback_model()

    @classmethod
    def _load_causal_model(cls, model_name):
        """Carga un modelo causal (GPT-2, BlenderBot, etc.)."""
        try:
            cls.main_tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.main_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Configurar padding
            if cls.main_tokenizer.pad_token is None:
                cls.main_tokenizer.pad_token = cls.main_tokenizer.eos_token
            cls.main_tokenizer.padding_side = 'left'
            
            logging.info(f"Modelo causal {model_name} cargado correctamente")
            
        except Exception as e:
            logging.error(f"Error al cargar modelo causal {model_name}: {e}")
            raise

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
    def _check_common_question(cls, message):
        """Verifica si la pregunta es una de las preguntas comunes predefinidas."""
        message_lower = message.lower().strip()
        
        # Verificar preguntas exactas
        for key, response in cls.common_questions.items():
            if key in message_lower:
                return response.strip()
        
        # Verificar patrones de preguntas sobre la ESPOCH
        if "qué es" in message_lower and "espoch" in message_lower:
            return cls.common_questions["qué es la espoch"].strip()
        
        if "historia" in message_lower and "espoch" in message_lower:
            return cls.common_questions["historia de la espoch"].strip()
        
        if "misión" in message_lower and "espoch" in message_lower:
            return cls.common_questions["misión de la espoch"].strip()
        
        if "visión" in message_lower and "espoch" in message_lower:
            return cls.common_questions["visión de la espoch"].strip()
        
        if "sede orellana" in message_lower or ("sede" in message_lower and "orellana" in message_lower):
            return cls.common_questions["espoch sede orellana"].strip()
        
        return None

    @classmethod
    def get_response(cls, message):
        """Genera la respuesta al mensaje del usuario."""
        logging.info(f"Recibido mensaje: {message}")
        try:
            if cls.connection is None or not cls.connection.is_connected():
                cls.initialize()

            # Guardar la pregunta para el feedback
            cls.last_question = message

            # Actualizar historial de conversación
            cls.conversation_history.append({"role": "user", "content": message})
            if len(cls.conversation_history) > 10:
                cls.conversation_history = cls.conversation_history[-10:]

            # 0. Buscar en la base de conocimiento aprendida
            knowledge_answer = knowledge_manager.get_answer(message, threshold=0.75)
            if knowledge_answer and knowledge_answer['confianza'] > 0.6:
                response = knowledge_answer['respuesta']
                cls.last_response = response
                cls.conversation_history.append({"role": "assistant", "content": response})
                logging.info(f"Respuesta encontrada en base de conocimiento (confianza: {knowledge_answer['confianza']:.2f})")
                return response

            # 1. Verificar si es una pregunta común predefinida
            common_response = cls._check_common_question(message)
            if common_response:
                cls.last_response = common_response
                cls.conversation_history.append({"role": "assistant", "content": common_response})
                return common_response

            # 2. Buscar intent directo en la base de datos (respuestas predefinidas)
            intent_response = cls.match_intent(message)
            if intent_response:
                cls.last_response = intent_response
                cls.conversation_history.append({"role": "assistant", "content": intent_response})
                return intent_response

            # 3. Verificar caché
            if message in cls.response_cache:
                logging.info("Respuesta encontrada en caché")
                cls.last_response = cls.response_cache[message]
                return cls.response_cache[message]

            # 4. Verificar si la pregunta está relacionada con ESPOCH
            if cls._is_espoch_related(message):
                # Usar contexto específico de ESPOCH y modelo GODEL
                context = cls.prepare_beca_ayuda_context()
                godel_response = cls._get_model_response(context, message)
                
                cls.last_response = godel_response
                cls.conversation_history.append({"role": "assistant", "content": godel_response})
                cls.response_cache[message] = godel_response
                return godel_response
            else:
                # Para preguntas no relacionadas, dar respuesta educada pero dirigida
                general_response = cls._get_general_response(message)
                cls.last_response = general_response
                cls.conversation_history.append({"role": "assistant", "content": general_response})
                cls.response_cache[message] = general_response
                return general_response
            
        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")    
            error_response = "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta de nuevo más tarde."
            cls.last_response = error_response
            return error_response

    @classmethod
    def _get_model_response(cls, context, question):
        """Genera respuesta usando el modelo principal configurado."""
        try:
            if cls.main_model is None or cls.main_tokenizer is None:
                return cls._get_qa_response(context, question)
            
            # Todos los modelos actuales son causales
            return cls._get_causal_response(context, question)
                
        except Exception as e:
            logging.error(f"Error en modelo principal: {str(e)}")
            return cls._get_qa_response(context, question)

    @classmethod
    def _get_causal_response(cls, context, question):
        """Genera respuesta usando modelos causales (GPT-2, BlenderBot, etc.)."""
        try:
            # Construir prompt conversacional optimizado
            if cls.CURRENT_MODEL == "blenderbot":
                # BlenderBot prefiere conversaciones más naturales
                prompt = f"""Eres PochiBot, un asistente amigable de la ESPOCH Sede Orellana.

Información: {context[:500]}

Usuario: {question}
PochiBot:"""
            else:
                # Para otros modelos GPT-2
                prompt = f"""Eres PochiBot, asistente virtual de la ESPOCH Sede Orellana. Ayudas con información sobre becas, carreras y servicios universitarios.

Contexto: {context[:600]}

Pregunta: {question}
Respuesta:"""

            inputs = cls.main_tokenizer.encode_plus(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = cls.main_model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,
                    num_beams=3,
                    temperature=0.8,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    pad_token_id=cls.main_tokenizer.pad_token_id
                )
            
            # Extraer solo la respuesta nueva
            response = cls.main_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            response = cls._clean_response(response, question)
            
            if response and len(response) > 15:
                return response
            else:
                return cls._get_qa_response(context, question)
                
        except Exception as e:
            logging.error(f"Error en modelo causal: {str(e)}")
            return cls._get_qa_response(context, question)

    @classmethod
    def _clean_response(cls, response, original_question):
        """Limpia y mejora la respuesta generada para que sea más organizada."""
        if not response:
            return None
        
        # Limpiar respuesta
        response = response.strip()
        
        # Remover fragmentos del prompt que puedan aparecer
        unwanted_phrases = [
            "Instruction:", "Knowledge:", "Dialogue:", "Estudiante:", "PochiBot:",
            "Eres PochiBot", "asistente virtual", "ESPOCH Sede Orellana"
        ]
        
        for phrase in unwanted_phrases:
            response = response.replace(phrase, "").strip()
        
        # Limpiar caracteres extraños
        response = response.replace("  ", " ").strip()
        
        # Verificar si la respuesta es demasiado corta o genérica
        if len(response) < 30 or response.lower() in [
            "ayudas económicas, carreras, servicios universitarios y trámites académicos.",
            "becas, ayudas económicas, carreras y servicios universitarios.",
            "te puedo ayudar con información sobre la espoch."
        ]:
            # Si la pregunta es sobre qué es la ESPOCH, dar respuesta predefinida
            if "qué es" in original_question.lower() and "espoch" in original_question.lower():
                return cls.common_questions["qué es la espoch"].strip()
            
            # Para otras preguntas cortas, intentar mejorar
            original_lower = original_question.lower()
            if "historia" in original_lower and "espoch" in original_lower:
                return cls.common_questions["historia de la espoch"].strip()
            elif "misión" in original_lower and "espoch" in original_lower:
                return cls.common_questions["misión de la espoch"].strip()
            elif "visión" in original_lower and "espoch" in original_lower:
                return cls.common_questions["visión de la espoch"].strip()
            elif "sede orellana" in original_lower:
                return cls.common_questions["espoch sede orellana"].strip()
        
        # Organizar la respuesta en formato estructurado
        # Detectar si hay puntos o elementos que puedan ser listados
        sentences = response.split('.')
        if len(sentences) > 2:
            # Si hay varias oraciones, intentar estructurarlas como lista
            structured_response = ""
            
            # Primera oración como introducción
            structured_response += sentences[0].strip() + ".\n\n"
            
            # Resto como puntos
            points = []
            for sentence in sentences[1:-1]:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    points.append(f"• {sentence.capitalize()}.")
            
            if points:
                structured_response += "\n".join(points)
                
                # Añadir conclusión si hay una última oración significativa
                if len(sentences[-1].strip()) > 10:
                    structured_response += f"\n\n{sentences[-1].strip().capitalize()}."
                
                # Añadir pregunta de seguimiento
                structured_response += "\n\n¿Hay algo más en lo que pueda ayudarte?"
                
                return structured_response
        
        # Si no se pudo estructurar, mejorar formato básico
        if "," in response and len(response) > 100:
            # Intentar convertir comas en puntos de lista
            parts = response.split(",")
            if len(parts) > 2:
                structured_response = parts[0].strip() + ":\n\n"
                for part in parts[1:]:
                    part = part.strip()
                    if part and len(part) > 5:
                        structured_response += f"• {part.capitalize()}\n"
                return structured_response
        
        # Verificar que no sea repetición de la pregunta
        if original_question.lower() in response.lower():
            return None
        
        return response

    @classmethod
    def _get_general_response(cls, message):
        """Genera respuesta estructurada para preguntas no relacionadas con ESPOCH."""
        message_lower = message.lower()
        
        # Respuestas para saludos generales
        if any(word in message_lower for word in ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos']):
            return ("¡Hola! Soy PochiBot, tu asistente virtual de la ESPOCH Sede Orellana.\n\n"
                   "Estoy aquí para ayudarte con:\n"
                   "• Información sobre becas y ayudas económicas\n"
                   "• Detalles sobre nuestras 5 carreras\n"
                   "• Procesos y trámites académicos\n"
                   "• Servicios universitarios disponibles\n\n"
                   "¿En qué tema específico puedo asistirte hoy?")
        
        # Respuestas para despedidas
        if any(word in message_lower for word in ['adiós', 'chao', 'hasta luego', 'nos vemos', 'gracias']):
            return ("¡Hasta luego! Fue un placer ayudarte.\n\n"
                   "Recuerda que estoy disponible para:\n"
                   "• Resolver tus dudas sobre la ESPOCH\n"
                   "• Brindarte información actualizada\n"
                   "• Orientarte en tus trámites académicos\n\n"
                   "¡Que tengas un excelente día!")
        
        # Respuestas para preguntas sobre el chatbot
        if any(word in message_lower for word in ['quién eres', 'qué eres', 'cómo te llamas', 'tu nombre']):
            return ("Soy PochiBot, el asistente virtual oficial de la ESPOCH Sede Orellana.\n\n"
                   "Mi función es:\n"
                   "• Ayudar a estudiantes con información precisa\n"
                   "• Orientar sobre becas y ayudas económicas\n"
                   "• Brindar detalles sobre nuestras carreras\n"
                   "• Facilitar información sobre servicios universitarios\n\n"
                   "¿Hay algo específico sobre la ESPOCH en lo que pueda ayudarte?")
        
        # Para otras preguntas generales, redirigir amablemente
        return ("Gracias por tu pregunta. Soy PochiBot, especializado en temas de la ESPOCH Sede Orellana.\n\n"
                "Puedo asistirte con información sobre:\n"
                "• Becas y ayudas económicas\n"
                "• Nuestras 5 carreras: Agronomía, Turismo, Ingeniería Ambiental, Zootecnia y Tecnologías de la Información\n"
                "• Requisitos y procesos de solicitud\n"
                "• Servicios universitarios\n"
                "• Información general del campus\n\n"
                "¿En qué área específica necesitas ayuda?")

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

            context = "INFORMACIÓN ESPOCH SEDE ORELLANA:\n"
            
            # Información sobre la ESPOCH (añadida para responder preguntas básicas)
            context += "SOBRE LA ESPOCH:\n"
            context += "La ESPOCH (Escuela Superior Politécnica de Chimborazo) es una institución pública de educación superior fundada el 18 de abril de 1969. "
            context += "Es una de las principales universidades técnicas de Ecuador, con su campus principal en Riobamba. "
            context += "La ESPOCH Sede Orellana es una extensión que ofrece educación superior de calidad en la región amazónica ecuatoriana. "
            context += "La institución está acreditada por el CACES y se destaca por su enfoque en carreras técnicas, ingenierías y ciencias aplicadas.\n\n"
            
            # Misión y Visión
            context += "MISIÓN Y VISIÓN:\n"
            context += "Misión: Formar profesionales e investigadores competentes, para contribuir al desarrollo sostenible del país.\n"
            context += "Visión: Ser una institución de educación superior líder en la Zona 3 del Ecuador, con reconocimiento nacional y proyección internacional.\n\n"
            
            # Información sobre becas y ayudas
            context += "BECAS Y AYUDAS ECONÓMICAS DISPONIBLES:\n"
            for row in data:
                context += f"• {row['tipo']} - {row['nombre']}: {row['descripcion']}\n"
            
            # Información específica de ESPOCH Sede Orellana
            context += "\nSOBRE LA ESPOCH SEDE ORELLANA:\n"
            context += "• Ubicación: Provincia de Orellana, Ecuador\n"
            context += "• Es una extensión de la Escuela Superior Politécnica de Chimborazo\n"
            context += "• Creada para ampliar la oferta académica en la Amazonía ecuatoriana\n"
            context += "• Brinda acceso a educación superior de calidad en la región\n"
            
            context += "\nCARRERAS DISPONIBLES:\n"
            context += "• Agronomía\n"
            context += "• Turismo\n"
            context += "• Ingeniería Ambiental\n"
            context += "• Zootecnia\n"
            context += "• Tecnologías de la Información\n"
            
            context += "\nSERVICIOS Y FACILIDADES:\n"
            context += "• Biblioteca especializada\n"
            context += "• Laboratorios de computación y ciencias\n"
            context += "• Departamento de Bienestar Estudiantil\n"
            context += "• Servicios médicos básicos\n"
            context += "• Áreas deportivas y recreativas\n"
            context += "• Comedor estudiantil\n"
            
            context += "\nPERÍODOS ACADÉMICOS:\n"
            context += "• Primer semestre: Septiembre a Febrero\n"
            context += "• Segundo semestre: Marzo a Agosto\n"
            
            context += "\nCONTACTO Y UBICACIÓN:\n"
            context += "• Para más información: Departamento de Bienestar Estudiantil\n"
            context += "• Ubicación: Edificio administrativo de la ESPOCH Sede Orellana\n"
            context += "• Los estudiantes pueden acercarse personalmente para consultas específicas\n"

            return context
        except Exception as e:
            logging.error(f"Error al preparar el contexto: {str(e)}")
            return "Error al obtener información sobre becas y ayudas económicas."

    @classmethod
    def _load_qa_model(cls):
        """Carga modelo de Question Answering en español."""
        try:
            qa_model_name = "mrm8488/bert-spanish-cased-finetuned-squad2-es"
            cls.qa_pipeline = pipeline(
                "question-answering",
                model=qa_model_name,
                tokenizer=qa_model_name,
                device=-1
            )
            logging.info("Modelo QA en español cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar modelo QA: {e}")
            # Fallback a modelo QA más simple
            try:
                cls.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=-1
                )
                logging.info("Modelo QA de respaldo cargado")
            except Exception as e2:
                logging.error(f"Error al cargar modelo QA de respaldo: {e2}")
                cls.qa_pipeline = None

    @classmethod
    def _load_fallback_model(cls):
        """Carga modelo de respaldo simple."""
        try:
            logging.info("Cargando modelo de respaldo...")
            cls.main_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            cls.main_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            
            if cls.main_tokenizer.pad_token is None:
                cls.main_tokenizer.pad_token = cls.main_tokenizer.eos_token
            cls.main_tokenizer.padding_side = 'left'
            
            cls.CURRENT_MODEL = "distilgpt2"
            logging.info("Modelo de respaldo cargado")
        except Exception as e:
            logging.error(f"Error al cargar modelo de respaldo: {e}")
            cls.main_model = None
            cls.main_tokenizer = None

    @classmethod
    def _get_qa_response(cls, context, question):
        """Genera una respuesta usando el modelo de Question Answering."""
        try:
            if cls.qa_pipeline is None:
                cls._load_models()
            
            # Verificar si es una pregunta común antes de usar QA
            common_response = cls._check_common_question(question)
            if common_response:
                return common_response
            
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
        """Maneja el feedback del usuario con sistema de autoaprendizaje."""
        try:
            # Sistema de autoaprendizaje basado en feedback
            if cls.last_question and cls.last_response:
                knowledge_manager.learn_from_conversation(
                    pregunta=cls.last_question,
                    respuesta_chatbot=cls.last_response,
                    feedback_usuario=feedback
                )
            
            # Respuestas al feedback
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
        cls.last_question = None
        cls.last_response = None
        return "Historial de conversación borrado."

    @classmethod
    def add_manual_knowledge(cls, pregunta, respuesta, categoria="becas"):
        """Permite añadir conocimiento manualmente al sistema."""
        try:
            success = knowledge_manager.add_knowledge(
                pregunta=pregunta,
                respuesta=respuesta,
                categoria=categoria,
                confianza=1.0,
                fuente="manual"
            )
            if success:
                return f"Conocimiento añadido exitosamente: {pregunta[:50]}..."
            else:
                return "No se pudo añadir el conocimiento. Puede que ya exista una pregunta similar."
        except Exception as e:
            logging.error(f"Error al añadir conocimiento manual: {e}")
            return "Error al añadir el conocimiento."

    @classmethod
    def get_knowledge_stats(cls):
        """Obtiene estadísticas de la base de conocimiento."""
        try:
            return knowledge_manager.get_knowledge_stats()
        except Exception as e:
            logging.error(f"Error al obtener estadísticas: {e}")
            return {}

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
