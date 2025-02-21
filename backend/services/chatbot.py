import os
import random
import logging
import mysql.connector
from mysql.connector import Error
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

class ChatbotService:
    connection = None
    model = None
    tokenizer = None
    conversation_history = []
    vectorizer = None
    qa_pipeline = None
    stop_words = set(stopwords.words('spanish'))
    beto_qa_pipeline = None
    torch.no_grad()  # Desactiva el cálculo de gradientes

    @classmethod
    def initialize(cls):
        """
        Inicializa la conexión a MySQL y carga el modelo de Hugging Face.
        """
        try:
            cls.connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST'),
                database=os.getenv('MYSQL_DATABASE'),
                user=os.getenv('MYSQL_USER'),
                password=os.getenv('MYSQL_PASSWORD'),
                port=os.getenv('MYSQL_PORT', 3306),
            )
            if cls.connection.is_connected():
                logging.info("Conectado a MySQL correctamente.")

            # Cargar modelo DistilBERT y tokenizer
            cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            cls.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-multilingual-cased")
            cls.qa_pipeline = pipeline("question-answering", model=cls.model, tokenizer=cls.tokenizer)
            cls.beto_qa_pipeline = pipeline("question-answering", model="dccuchile/bert-base-spanish-wwm-cased", tokenizer="dccuchile/bert-base-spanish-wwm-cased")

            logging.info("Modelo DistilBERT multilingüe cargado correctamente.")

            # Inicializar el vectorizador para la detección de intenciones
            cls.vectorizer = TfidfVectorizer(stop_words=list(cls.stop_words))
            all_intents = cls.get_all_intents()
            cls.vectorizer.fit(all_intents)

        except Error as e:
            logging.error(f"Error al conectar a MySQL: {e}")
            raise
        except Exception as e:
            logging.error(f"Error al cargar datos o modelo: {str(e)}")
            raise

    @classmethod
    def get_all_intents(cls):
        """
        Obtiene todas las preguntas de intents de la base de datos.
        """
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

    @classmethod
    def match_intent(cls, message):
        """
        Busca un intent correspondiente al mensaje del usuario en la base de datos.
        """
        cursor = cls.connection.cursor(dictionary=True)
        query = """
        SELECT 'beca' as tipo, rb.respuesta
        FROM preguntas_beca pb
        JOIN intents_beca ib ON pb.intent_beca_id = ib.id
        JOIN respuestas_beca rb ON rb.intent_beca_id = ib.id
        WHERE LOWER(pb.pregunta) LIKE %s
        UNION
        SELECT 'ayuda' as tipo, ra.respuesta
        FROM preguntas_ayudas pa
        JOIN intents_ayudas ia ON pa.intent_id = ia.id
        JOIN respuestas_ayudas ra ON ra.intent_id = ia.id
        WHERE LOWER(pa.pregunta) LIKE %s
        UNION
        SELECT 'saludo' as tipo, rs.respuesta
        FROM preguntas_saludo ps
        JOIN intents_saludo isa ON ps.intent_saludo_id = isa.id
        JOIN respuestas_saludo rs ON rs.intent_saludo_id = isa.id
        WHERE LOWER(ps.pregunta) LIKE %s
        """
        cursor.execute(query, (f"%{message.lower()}%", f"%{message.lower()}%", f"%{message.lower()}%"))
        results = cursor.fetchall()
        cursor.close()

        if results:
            return random.choice(results)
        return None

    @classmethod
    def is_beca_related(cls, message):
        """
        Determina si el mensaje está relacionado con becas o ayudas económicas.
        """
        beca_keywords = ['beca', 'ayuda', 'económica', 'financiamiento', 'estudios', 'universidad', 'ESPOCH']
        return any(keyword in message.lower() for keyword in beca_keywords)

    @classmethod
    def get_bert_response(cls, context, question):
        """
        Genera una respuesta usando el modelo DistilBERT.
        """
        try:
            # Dividir el contexto en chunks más pequeños si es muy largo
            max_length = 512
            context_chunks = [context[i:i+max_length] for i in range(0, len(context), max_length)]
            
            best_answer = ""
            best_score = 0
            
            for chunk in context_chunks:
                result = cls.qa_pipeline(question=question, context=chunk)
                if result['score'] > best_score:
                    best_answer = result['answer']
                    best_score = result['score']
            
            if best_answer.strip() == '' or best_answer.strip().lower() == '[cls]' or best_score < 0.1:
                return "Lo siento, no tengo suficiente información para responder a esa pregunta específica. ¿Podrías reformularla o preguntar sobre algo más general relacionado con las becas o ayudas económicas?"
            
            return best_answer.strip()
        except Exception as e:
            logging.error(f"Error al generar respuesta con DistilBERT: {str(e)}")
            return "Lo siento, ocurrió un error al procesar tu pregunta. ¿Podrías intentar reformularla?"

    @classmethod
    def get_response(cls, message):
        """
        Genera la respuesta al mensaje del usuario.
        """
        if cls.connection is None or cls.model is None:
            cls.initialize()

        try:
            # Agregar el mensaje del usuario al historial de conversación
            cls.conversation_history.append({"role": "user", "content": message})

            # Verificar si es un escenario de despedida
            farewell_response = cls.handle_farewell(message)
            if farewell_response:
                cls.conversation_history.append({"role": "assistant", "content": farewell_response})
                return farewell_response

            # Buscar respuesta en intents
            intent_match = cls.match_intent(message)
            if intent_match:
                response = intent_match['respuesta']
                cls.conversation_history.append({"role": "assistant", "content": response})
                return response

            # Determinar si la pregunta está relacionada con becas o ayudas económicas
            is_beca_related = cls.is_beca_related(message)
            logging.info(f"¿La pregunta está relacionada con becas o ayudas económicas? {is_beca_related}")

            if is_beca_related:
                context = cls.prepare_beca_ayuda_context()
                bert_response = cls.get_bert_response(context, message)
            
                # Añadir información adicional si la respuesta es corta
                if len(bert_response.split()) < 10:
                    bert_response += " " + cls.get_additional_info(message)
            else:
                bert_response = cls.get_beto_response(message)

            cls.conversation_history.append({"role": "assistant", "content": bert_response})
            return bert_response

        except Exception as e:
            logging.error(f"Error al generar respuesta: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu solicitud. ¿Puedes intentar reformular tu pregunta?"

    @classmethod
    def detect_intent(cls, message):
        """
        Detecta la intención del usuario basándose en su mensaje.
        """
        message_vector = cls.vectorizer.transform([message])
        all_intents = cls.get_all_intents()
        intent_vectors = cls.vectorizer.transform(all_intents)
        similarities = cosine_similarity(message_vector, intent_vectors)
        most_similar_index = similarities.argmax()
        return all_intents[most_similar_index]

    @classmethod
    def prepare_beca_ayuda_context(cls):
        """
        Prepara el contexto de becas y ayudas económicas para la generación de respuestas.
        """
        cursor = cls.connection.cursor(dictionary=True)
        beca_query = """
        SELECT b.nombre, b.descripcion, GROUP_CONCAT(r.requisito SEPARATOR '. ') as requisitos
        FROM becas b
        LEFT JOIN requisitos_beca r ON b.id = r.beca_id
        GROUP BY b.id
        """
        ayuda_query = """
        SELECT nombre, descripcion, requisitos, duracion
        FROM ayudas_economicas
        """
        cursor.execute(beca_query)
        becas_data = cursor.fetchall()
        cursor.execute(ayuda_query)
        ayudas_data = cursor.fetchall()
        cursor.close()

        context = "Información sobre becas y ayudas económicas disponibles en la ESPOCH: "
        for beca in becas_data:
            context += f"Beca {beca['nombre']}: {beca['descripcion']} Requisitos: {beca['requisitos']}. "
        for ayuda in ayudas_data:
            context += f"Ayuda económica {ayuda['nombre']}: {ayuda['descripcion']} Requisitos: {ayuda['requisitos']}. Duración: {ayuda['duracion']}. "
        
        return context

    @classmethod
    def prepare_general_context(cls):
        """
        Prepara un contexto general para preguntas no relacionadas con becas o ayudas económicas.
        """
        return """
        Soy un asistente virtual diseñado para proporcionar información sobre becas, ayudas económicas y servicios estudiantiles 
        en la Escuela Superior Politécnica de Chimborazo (ESPOCH). Puedo ayudar con información sobre tipos de becas y ayudas, 
        requisitos, procesos de aplicación y fechas importantes. También puedo proporcionar información general 
        sobre la universidad y sus servicios. Si tienes preguntas específicas sobre becas, ayudas económicas o la ESPOCH, no dudes en preguntar.
        """

    @classmethod
    def get_additional_info(cls, message):
        """
        Proporciona información adicional basada en palabras clave en el mensaje.
        """
        keywords = {
            'beca': 'Las becas son ayudas económicas para estudiantes. La ESPOCH ofrece varios tipos de becas.',
            'ayuda económica': 'Las ayudas económicas son apoyos financieros para estudiantes en situaciones específicas.',
            'requisito': 'Los requisitos varían según el tipo de beca o ayuda económica. Generalmente incluyen buen rendimiento académico y situación económica.',
            'fecha': 'Las fechas de aplicación para becas y ayudas económicas varían cada semestre. Te recomiendo consultar la página oficial de la ESPOCH para las fechas más actualizadas.',
            'proceso': 'El proceso de aplicación generalmente implica llenar un formulario y presentar documentos que respalden tu solicitud.',
            'ESPOCH': 'La ESPOCH es una universidad pública ubicada en Riobamba, Ecuador, conocida por su excelencia académica.'
        }
        
        for key, info in keywords.items():
            if key in message.lower():
                return info
        
        return "Para más información, te recomiendo visitar la página oficial de la ESPOCH o contactar directamente con la oficina de becas y ayudas económicas."

    @classmethod
    def handle_feedback(cls, feedback, last_response):
        """
        Maneja la retroalimentación del usuario sobre la última respuesta.
        """
        if feedback.lower() == 'útil':
            logging.info(f"Respuesta útil: {last_response}")
            return "¡Me alegra haber sido de ayuda! ¿Hay algo más en lo que pueda asistirte?"
        else:
            logging.info(f"Respuesta no útil: {last_response}")
            return "Lamento no haber sido de ayuda. ¿Podrías proporcionar más detalles sobre tu pregunta para que pueda intentar darte una mejor respuesta?"

    @classmethod
    def get_sentiment(cls, text):
        """
        Analiza el sentimiento del texto del usuario.
        """
        positive_words = ['gracias', 'excelente', 'bueno', 'genial', 'útil']
        negative_words = ['malo', 'inútil', 'terrible', 'confuso', 'difícil']
        
        words = word_tokenize(text.lower())
        sentiment_score = sum(1 for word in words if word in positive_words) - sum(1 for word in words if word in negative_words)
        
        if sentiment_score > 0:
            return "positivo"
        elif sentiment_score < 0:
            return "negativo"
        else:
            return "neutral"

    @classmethod
    def handle_farewell(cls, message):
        """
        Maneja escenarios donde el usuario expresa que no desea más información sobre becas o ayudas económicas.
        """
        farewell_keywords = ['no quiero', 'no deseo', 'no necesito', 'es suficiente', 'gracias', 'adiós', 'chao']
        if any(keyword in message.lower() for keyword in farewell_keywords):
            farewell_responses = [
                "Entiendo. Si en el futuro necesitas información sobre becas o ayudas económicas, no dudes en volver a consultarme. ¡Que tengas un buen día!",
                "De acuerdo. Recuerda que estoy aquí para ayudarte con cualquier duda sobre becas o ayudas económicas cuando lo necesites. ¡Hasta luego!",
                "Perfecto. Si más adelante tienes preguntas sobre becas, ayudas económicas o cualquier otro tema relacionado con la ESPOCH, estaré encantado de ayudarte. ¡Cuídate!",
                "Muy bien. Recuerda que la información sobre becas y ayudas económicas está siempre disponible si la necesitas en el futuro. ¡Que te vaya bien!",
                "Entendido. Si en algún momento necesitas más información o tienes otras preguntas sobre la ESPOCH, no dudes en volver. ¡Hasta la próxima!"
            ]
            return random.choice(farewell_responses)
        return None

    @classmethod
    def get_ai_response(cls, message):
        """
        Genera una respuesta usando el modelo de IA para preguntas no relacionadas con becas o ayudas económicas.
        """
        try:
            # Usar el modelo DistilBERT para generar una respuesta
            context = "Eres un asistente virtual amigable y útil. Responde de manera concisa y apropiada a la pregunta del usuario."
            response = cls.get_bert_response(context, message)
            
            # Asegurarse de que la respuesta no sea sobre becas o ayudas económicas de la ESPOCH
            if cls.is_beca_related(response):
                return "Lo siento, no tengo información específica sobre ese tema. ¿Hay algo más en lo que pueda ayudarte?"
            
            return response
        except Exception as e:
            logging.error(f"Error al generar respuesta con IA: {str(e)}")
            return "Lo siento, no pude procesar esa pregunta. ¿Podrías reformularla o preguntar sobre algo más?"
    
    @classmethod
    def get_beto_response(cls, message):
        try:
            # Contexto general sobre la ESPOCH y las funciones del chatbot
            context = """
            La ESPOCH (Escuela Superior Politécnica de Chimborazo) es una prestigiosa universidad pública ubicada en Riobamba, Ecuador. Fundada en 1972, se destaca por su excelencia académica e investigación innovadora. Ofrece una amplia gama de programas en ingeniería, ciencias, administración y tecnología. La universidad cuenta con modernos laboratorios, bibliotecas bien equipadas y diversos servicios estudiantiles. Fomenta la participación en proyectos de investigación, programas de intercambio internacional y actividades extracurriculares para enriquecer la experiencia universitaria.

            Como asistente virtual especializado en la ESPOCH, puedo:
            1. Proporcionar información detallada sobre becas y ayudas económicas disponibles.
            2. Explicar requisitos y procesos de aplicación para becas y ayudas.
            3. Ofrecer información sobre programas académicos, instalaciones y servicios de la ESPOCH.
            4. Responder preguntas sobre la vida estudiantil y trámites administrativos.
            5. Ayudar con consultas generales sobre la universidad y su funcionamiento.

            Utilizo tecnología de procesamiento de lenguaje natural y una base de datos con información actualizada sobre la ESPOCH para proporcionar respuestas precisas y útiles a tus preguntas.
            """
        
            # Generar respuesta utilizando el modelo BETO
            result = cls.beto_qa_pipeline(question=message, context=context)
            response = result['answer']

            # Si la respuesta es muy corta o tiene baja confianza, generar una respuesta más elaborada
            if len(response.split()) < 10 or result['score'] < 0.3:
                elaborated_question = f"Proporciona una respuesta detallada y útil a la siguiente pregunta sobre la ESPOCH: {message}"
                elaborated_result = cls.beto_qa_pipeline(question=elaborated_question, context=context)
                response = elaborated_result['answer']

            # Limpiar la respuesta
            response = response.strip()
            if response.lower().startswith("explicar"):
                response = response[8:].strip()  # Eliminar "Explicar" y espacios en blanco

            # Si la respuesta sigue siendo corta o irrelevante, usar una respuesta predeterminada
            if len(response.split()) < 10 or "no tengo información" in response.lower():
                response = """
                La ESPOCH (Escuela Superior Politécnica de Chimborazo) es una prestigiosa universidad pública ubicada en Riobamba, Ecuador. 
                Fundada en 1972, se destaca por su excelencia académica, investigación innovadora y compromiso con el desarrollo sostenible. 
                Ofrece una amplia gama de programas en áreas como ingeniería, ciencias, administración y tecnología. 
                La universidad cuenta con modernos laboratorios, bibliotecas bien equipadas y diversos servicios estudiantiles para apoyar el aprendizaje y desarrollo de sus estudiantes.
                """

            # Añadir una frase introductoria para hacer la respuesta más natural
            introductions = [
                "Basándome en la información disponible, ",
                "Según mi conocimiento sobre la ESPOCH, ",
                "Analizando tu pregunta, puedo decirte que ",
                "Considerando el contexto de la ESPOCH, ",
                "En respuesta a tu consulta, "
            ]

            response = random.choice(introductions) + response
        
            return response

        except Exception as e:
            logging.error(f"Error al generar respuesta con BETO: {str(e)}")
            return "Lo siento, tuve un problema al procesar esa pregunta. ¿Podrías intentar reformularla o preguntar sobre otro aspecto de la ESPOCH o mis funciones?"


if __name__ == "__main__":
    chatbot = ChatbotService()
    print("Chatbot: Soy PochiBot, tu asistente virtual para información sobre becas y ayudas económicas en la ESPOCH. ¿En qué puedo ayudarte hoy?")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['salir', 'adiós', 'chao']:
            print("Chatbot: ¡Hasta luego! Espero haber sido de ayuda.")
            break
        
        sentiment = chatbot.get_sentiment(user_input)
        response = chatbot.get_response(user_input)
        
        if sentiment == "positivo":
            response = "Me alegra que estés satisfecho. " + response
        elif sentiment == "negativo":
            response = "Entiendo tu preocupación. Intentaré explicarlo mejor. " + response
        
        print(f"Chatbot: {response}")
        print("¿Fue útil esta respuesta? (Responde 'útil' o 'no útil')")
        feedback = input("Tu opinión: ")
        feedback_response = chatbot.handle_feedback(feedback, response)
        print(f"Chatbot: {feedback_response}")

