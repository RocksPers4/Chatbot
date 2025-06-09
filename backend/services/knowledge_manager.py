import os
import json
import logging
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeManager:
    """
    Gestor de conocimiento para el chatbot de ESPOCH.
    Permite almacenar, buscar y gestionar preguntas y respuestas sobre becas y ayudas económicas.
    """
    
    def __init__(self):
        self.connection = None
        self.vectorizer = TfidfVectorizer(stop_words='spanish')
        self.knowledge_base = []
        self.question_vectors = None
        
        # Archivo local para respaldo del conocimiento
        self.knowledge_file = "data/knowledge_base.json"
        
        # Crear directorio si no existe
        os.makedirs("data", exist_ok=True)
        
        self.initialize()

    def initialize(self):
        """Inicializa la conexión a la base de datos y carga el conocimiento existente."""
        try:
            self.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                port=Config.MYSQL_PORT
            )
            if self.connection.is_connected():
                logging.info("KnowledgeManager: Conectado a MySQL correctamente.")
            
            # Crear tabla si no existe
            self._create_knowledge_table()
            
            # Cargar conocimiento existente
            self.load_knowledge_base()
            
        except mysql.connector.Error as e:
            logging.error(f"Error al conectar a MySQL en KnowledgeManager: {e}")
            # Cargar desde archivo local como respaldo
            self.load_from_file()

    def _create_knowledge_table(self):
        """Crea la tabla de conocimiento si no existe."""
        try:
            cursor = self.connection.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INT AUTO_INCREMENT PRIMARY KEY,
                pregunta TEXT NOT NULL,
                respuesta TEXT NOT NULL,
                categoria VARCHAR(100) DEFAULT 'general',
                confianza FLOAT DEFAULT 1.0,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                activo BOOLEAN DEFAULT TRUE,
                fuente VARCHAR(100) DEFAULT 'usuario',
                validado BOOLEAN DEFAULT FALSE
            )
            """
            cursor.execute(create_table_query)
            self.connection.commit()
            cursor.close()
            logging.info("Tabla knowledge_base verificada/creada correctamente.")
        except Exception as e:
            logging.error(f"Error al crear tabla knowledge_base: {e}")

    def add_knowledge(self, pregunta, respuesta, categoria="becas", confianza=1.0, fuente="usuario"):
        """
        Añade nuevo conocimiento a la base de datos.
        
        Args:
            pregunta (str): La pregunta del usuario
            respuesta (str): La respuesta correspondiente
            categoria (str): Categoría del conocimiento (becas, ayudas, general)
            confianza (float): Nivel de confianza en la respuesta (0.0 - 1.0)
            fuente (str): Fuente del conocimiento (usuario, admin, sistema)
        """
        try:
            # Verificar si ya existe una pregunta similar
            similar_question = self.find_similar_question(pregunta)
            if similar_question and similar_question['similarity'] > 0.8:
                logging.info(f"Pregunta similar ya existe: {similar_question['pregunta']}")
                return False
            
            cursor = self.connection.cursor()
            insert_query = """
            INSERT INTO knowledge_base (pregunta, respuesta, categoria, confianza, fuente)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (pregunta, respuesta, categoria, confianza, fuente))
            self.connection.commit()
            cursor.close()
            
            # Actualizar la base de conocimiento en memoria
            self.load_knowledge_base()
            
            # Guardar respaldo en archivo
            self.save_to_file()
            
            logging.info(f"Conocimiento añadido: {pregunta[:50]}...")
            return True
            
        except Exception as e:
            logging.error(f"Error al añadir conocimiento: {e}")
            return False

    def find_similar_question(self, pregunta, threshold=0.7):
        """
        Busca preguntas similares en la base de conocimiento.
        
        Args:
            pregunta (str): La pregunta a buscar
            threshold (float): Umbral de similitud mínimo
            
        Returns:
            dict: Información de la pregunta más similar o None
        """
        try:
            if not self.knowledge_base or not pregunta.strip():
                return None
            
            # Vectorizar la pregunta de entrada
            pregunta_vector = self.vectorizer.transform([pregunta.lower()])
            
            # Calcular similitudes
            similarities = cosine_similarity(pregunta_vector, self.question_vectors).flatten()
            
            # Encontrar la más similar
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= threshold:
                similar_item = self.knowledge_base[max_similarity_idx].copy()
                similar_item['similarity'] = float(max_similarity)
                return similar_item
            
            return None
            
        except Exception as e:
            logging.error(f"Error al buscar pregunta similar: {e}")
            return None

    def get_answer(self, pregunta, threshold=0.7):
        """
        Obtiene la respuesta para una pregunta específica.
        
        Args:
            pregunta (str): La pregunta del usuario
            threshold (float): Umbral de similitud mínimo
            
        Returns:
            dict: Respuesta encontrada o None
        """
        similar_question = self.find_similar_question(pregunta, threshold)
        if similar_question:
            return {
                'respuesta': similar_question['respuesta'],
                'confianza': similar_question['confianza'],
                'similarity': similar_question['similarity'],
                'categoria': similar_question['categoria']
            }
        return None

    def load_knowledge_base(self):
        """Carga toda la base de conocimiento desde la base de datos."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT * FROM knowledge_base 
            WHERE activo = TRUE 
            ORDER BY confianza DESC, fecha_actualizacion DESC
            """
            cursor.execute(query)
            self.knowledge_base = cursor.fetchall()
            cursor.close()
            
            # Crear vectores para búsqueda rápida
            if self.knowledge_base:
                questions = [item['pregunta'].lower() for item in self.knowledge_base]
                self.question_vectors = self.vectorizer.fit_transform(questions)
            
            logging.info(f"Base de conocimiento cargada: {len(self.knowledge_base)} elementos")
            
        except Exception as e:
            logging.error(f"Error al cargar base de conocimiento: {e}")
            self.knowledge_base = []

    def update_knowledge(self, pregunta_id, nueva_respuesta=None, nueva_confianza=None):
        """
        Actualiza conocimiento existente.
        
        Args:
            pregunta_id (int): ID de la pregunta a actualizar
            nueva_respuesta (str): Nueva respuesta (opcional)
            nueva_confianza (float): Nueva confianza (opcional)
        """
        try:
            cursor = self.connection.cursor()
            
            updates = []
            values = []
            
            if nueva_respuesta:
                updates.append("respuesta = %s")
                values.append(nueva_respuesta)
            
            if nueva_confianza is not None:
                updates.append("confianza = %s")
                values.append(nueva_confianza)
            
            if updates:
                values.append(pregunta_id)
                update_query = f"""
                UPDATE knowledge_base 
                SET {', '.join(updates)}, fecha_actualizacion = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(update_query, values)
                self.connection.commit()
                
                # Recargar base de conocimiento
                self.load_knowledge_base()
                self.save_to_file()
                
                logging.info(f"Conocimiento actualizado: ID {pregunta_id}")
            
            cursor.close()
            
        except Exception as e:
            logging.error(f"Error al actualizar conocimiento: {e}")

    def validate_knowledge(self, pregunta_id, validado=True):
        """
        Marca conocimiento como validado por un administrador.
        
        Args:
            pregunta_id (int): ID de la pregunta
            validado (bool): Estado de validación
        """
        try:
            cursor = self.connection.cursor()
            update_query = """
            UPDATE knowledge_base 
            SET validado = %s, fecha_actualizacion = CURRENT_TIMESTAMP
            WHERE id = %s
            """
            cursor.execute(update_query, (validado, pregunta_id))
            self.connection.commit()
            cursor.close()
            
            self.load_knowledge_base()
            logging.info(f"Conocimiento {'validado' if validado else 'invalidado'}: ID {pregunta_id}")
            
        except Exception as e:
            logging.error(f"Error al validar conocimiento: {e}")

    def get_unvalidated_knowledge(self):
        """Obtiene conocimiento no validado para revisión."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT * FROM knowledge_base 
            WHERE validado = FALSE AND activo = TRUE
            ORDER BY fecha_creacion DESC
            """
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return result
            
        except Exception as e:
            logging.error(f"Error al obtener conocimiento no validado: {e}")
            return []

    def get_knowledge_stats(self):
        """Obtiene estadísticas de la base de conocimiento."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            stats_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN validado = TRUE THEN 1 ELSE 0 END) as validados,
                SUM(CASE WHEN activo = TRUE THEN 1 ELSE 0 END) as activos,
                AVG(confianza) as confianza_promedio,
                COUNT(DISTINCT categoria) as categorias
            FROM knowledge_base
            """
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            
            # Estadísticas por categoría
            category_query = """
            SELECT categoria, COUNT(*) as cantidad
            FROM knowledge_base 
            WHERE activo = TRUE
            GROUP BY categoria
            ORDER BY cantidad DESC
            """
            cursor.execute(category_query)
            categories = cursor.fetchall()
            
            cursor.close()
            
            return {
                'general': stats,
                'por_categoria': categories
            }
            
        except Exception as e:
            logging.error(f"Error al obtener estadísticas: {e}")
            return {}

    def save_to_file(self):
        """Guarda la base de conocimiento en un archivo JSON como respaldo."""
        try:
            # Convertir datetime a string para JSON
            knowledge_for_json = []
            for item in self.knowledge_base:
                item_copy = item.copy()
                if 'fecha_creacion' in item_copy:
                    item_copy['fecha_creacion'] = str(item_copy['fecha_creacion'])
                if 'fecha_actualizacion' in item_copy:
                    item_copy['fecha_actualizacion'] = str(item_copy['fecha_actualizacion'])
                knowledge_for_json.append(item_copy)
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_for_json, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Base de conocimiento guardada en {self.knowledge_file}")
            
        except Exception as e:
            logging.error(f"Error al guardar archivo de conocimiento: {e}")

    def load_from_file(self):
        """Carga la base de conocimiento desde archivo como respaldo."""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                
                # Crear vectores para búsqueda
                if self.knowledge_base:
                    questions = [item['pregunta'].lower() for item in self.knowledge_base]
                    self.vectorizer = TfidfVectorizer(stop_words='spanish')
                    self.question_vectors = self.vectorizer.fit_transform(questions)
                
                logging.info(f"Base de conocimiento cargada desde archivo: {len(self.knowledge_base)} elementos")
            
        except Exception as e:
            logging.error(f"Error al cargar desde archivo: {e}")
            self.knowledge_base = []

    def learn_from_conversation(self, pregunta, respuesta_chatbot, feedback_usuario, confianza_base=0.7):
        """
        Aprende de las conversaciones basándose en el feedback del usuario.
        
        Args:
            pregunta (str): Pregunta del usuario
            respuesta_chatbot (str): Respuesta que dio el chatbot
            feedback_usuario (str): Feedback del usuario (positivo/negativo)
            confianza_base (float): Confianza base para el aprendizaje
        """
        try:
            # Determinar confianza basada en feedback
            if feedback_usuario.lower() in ['bueno', 'útil', 'correcto', 'gracias', 'excelente']:
                confianza = min(confianza_base + 0.2, 1.0)
                # Añadir conocimiento positivo
                self.add_knowledge(
                    pregunta=pregunta,
                    respuesta=respuesta_chatbot,
                    categoria=self._classify_question(pregunta),
                    confianza=confianza,
                    fuente="aprendizaje_positivo"
                )
                logging.info(f"Aprendizaje positivo registrado para: {pregunta[:50]}...")
                
            elif feedback_usuario.lower() in ['malo', 'incorrecto', 'no útil', 'error']:
                # Marcar como conocimiento a revisar (baja confianza)
                self.add_knowledge(
                    pregunta=pregunta,
                    respuesta="[RESPUESTA A REVISAR] " + respuesta_chatbot,
                    categoria=self._classify_question(pregunta),
                    confianza=0.1,
                    fuente="aprendizaje_negativo"
                )
                logging.info(f"Aprendizaje negativo registrado para revisión: {pregunta[:50]}...")
            
        except Exception as e:
            logging.error(f"Error en aprendizaje de conversación: {e}")

    def _classify_question(self, pregunta):
        """Clasifica automáticamente una pregunta en categorías."""
        pregunta_lower = pregunta.lower()
        
        if any(word in pregunta_lower for word in ['beca', 'becas']):
            return 'becas'
        elif any(word in pregunta_lower for word in ['ayuda económica', 'ayudas económicas', 'financiamiento']):
            return 'ayudas_economicas'
        elif any(word in pregunta_lower for word in ['carrera', 'carreras', 'estudios']):
            return 'carreras'
        elif any(word in pregunta_lower for word in ['matrícula', 'inscripción', 'trámite']):
            return 'tramites'
        elif any(word in pregunta_lower for word in ['horario', 'calendario', 'fecha']):
            return 'horarios'
        else:
            return 'general'

    def export_knowledge(self, filename=None):
        """Exporta la base de conocimiento a un archivo."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/knowledge_export_{timestamp}.json"
        
        try:
            self.save_to_file()
            # Copiar archivo con nuevo nombre
            import shutil
            shutil.copy2(self.knowledge_file, filename)
            logging.info(f"Base de conocimiento exportada a: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error al exportar conocimiento: {e}")
            return None

    def import_knowledge(self, filename):
        """Importa conocimiento desde un archivo."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                imported_knowledge = json.load(f)
            
            imported_count = 0
            for item in imported_knowledge:
                if self.add_knowledge(
                    pregunta=item['pregunta'],
                    respuesta=item['respuesta'],
                    categoria=item.get('categoria', 'general'),
                    confianza=item.get('confianza', 0.8),
                    fuente="importacion"
                ):
                    imported_count += 1
            
            logging.info(f"Conocimiento importado: {imported_count} elementos desde {filename}")
            return imported_count
            
        except Exception as e:
            logging.error(f"Error al importar conocimiento: {e}")
            return 0

# Instancia global del gestor de conocimiento
knowledge_manager = KnowledgeManager()

if __name__ == "__main__":
    # Ejemplo de uso
    km = KnowledgeManager()
    
    # Añadir conocimiento de ejemplo
    ejemplos_becas = [
        {
            "pregunta": "¿Cuáles son los requisitos para la beca de excelencia académica?",
            "respuesta": "Para la beca de excelencia académica necesitas: promedio mínimo de 8.5, no tener materias perdidas, presentar certificado de notas, y carta de motivación.",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Cuándo abren las convocatorias de becas?",
            "respuesta": "Las convocatorias de becas se abren generalmente en marzo para el primer semestre y en agosto para el segundo semestre. Te recomiendo estar atento a las publicaciones oficiales.",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Qué ayudas económicas hay disponibles?",
            "respuesta": "Tenemos varias ayudas económicas: beca alimentaria, ayuda de transporte, beca de materiales, y programa de trabajo estudiantil. Cada una tiene requisitos específicos.",
            "categoria": "ayudas_economicas"
        }
    ]
    
    for ejemplo in ejemplos_becas:
        km.add_knowledge(**ejemplo)
    
    # Probar búsqueda
    resultado = km.get_answer("requisitos beca excelencia")
    if resultado:
        print(f"Respuesta encontrada: {resultado['respuesta']}")
        print(f"Confianza: {resultado['confianza']}")
        print(f"Similitud: {resultado['similarity']}")
    
    # Mostrar estadísticas
    stats = km.get_knowledge_stats()
    print(f"Estadísticas: {stats}")