import os
import json
import logging
import mysql.connector
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config

logging.basicConfig(level=logging.INFO)

class KnowledgeManager:
    def __init__(self):
        self.connection = None
        self.vectorizer = TfidfVectorizer(stop_words='spanish')
        self.knowledge_base = []
        self.question_vectors = None
        self.knowledge_file = "data/knowledge_base.json"
        
        os.makedirs("data", exist_ok=True)
        self.initialize()

    def initialize(self):
        """Inicializa conexión y carga conocimiento."""
        try:
            self.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                port=Config.MYSQL_PORT
            )
            self._create_knowledge_table()
            self.load_knowledge_base()
        except Exception as e:
            logging.error(f"Error en KnowledgeManager: {e}")
            self.load_from_file()

    def _create_knowledge_table(self):
        """Crea tabla de conocimiento."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pregunta TEXT NOT NULL,
                    respuesta TEXT NOT NULL,
                    categoria VARCHAR(100) DEFAULT 'general',
                    confianza FLOAT DEFAULT 1.0,
                    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    activo BOOLEAN DEFAULT TRUE,
                    fuente VARCHAR(100) DEFAULT 'usuario',
                    validado BOOLEAN DEFAULT FALSE
                )
            """)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            logging.error(f"Error creando tabla: {e}")

    def add_knowledge(self, pregunta, respuesta, categoria="becas", confianza=1.0, fuente="usuario"):
        """Añade conocimiento nuevo."""
        try:
            # Verificar duplicados
            similar = self.find_similar_question(pregunta)
            if similar and similar['similarity'] > 0.8:
                return False
            
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO knowledge_base (pregunta, respuesta, categoria, confianza, fuente)
                VALUES (%s, %s, %s, %s, %s)
            """, (pregunta, respuesta, categoria, confianza, fuente))
            self.connection.commit()
            cursor.close()
            
            self.load_knowledge_base()
            self.save_to_file()
            return True
        except Exception as e:
            logging.error(f"Error añadiendo conocimiento: {e}")
            return False

    def find_similar_question(self, pregunta, threshold=0.7):
        """Busca preguntas similares."""
        try:
            if not self.knowledge_base or not pregunta.strip():
                return None
            
            pregunta_vector = self.vectorizer.transform([pregunta.lower()])
            similarities = cosine_similarity(pregunta_vector, self.question_vectors).flatten()
            
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity >= threshold:
                result = self.knowledge_base[max_idx].copy()
                result['similarity'] = float(max_similarity)
                return result
            return None
        except Exception as e:
            logging.error(f"Error buscando similitud: {e}")
            return None

    def get_answer(self, pregunta, threshold=0.7):
        """Obtiene respuesta para una pregunta."""
        similar = self.find_similar_question(pregunta, threshold)
        if similar:
            return {
                'respuesta': similar['respuesta'],
                'confianza': similar['confianza'],
                'similarity': similar['similarity'],
                'categoria': similar['categoria']
            }
        return None

    def load_knowledge_base(self):
        """Carga conocimiento desde BD."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM knowledge_base WHERE activo = TRUE ORDER BY confianza DESC")
            self.knowledge_base = cursor.fetchall()
            cursor.close()
            
            if self.knowledge_base:
                questions = [item['pregunta'].lower() for item in self.knowledge_base]
                self.question_vectors = self.vectorizer.fit_transform(questions)
        except Exception as e:
            logging.error(f"Error cargando conocimiento: {e}")
            self.knowledge_base = []

    def learn_from_conversation(self, pregunta, respuesta_chatbot, feedback_usuario, confianza_base=0.7):
        """Aprende de conversaciones basado en feedback."""
        try:
            if feedback_usuario.lower() in ['bueno', 'útil', 'correcto', 'gracias', 'excelente']:
                confianza = min(confianza_base + 0.2, 1.0)
                self.add_knowledge(pregunta, respuesta_chatbot, self._classify_question(pregunta), 
                                 confianza, "aprendizaje_positivo")
            elif feedback_usuario.lower() in ['malo', 'incorrecto', 'no útil', 'error']:
                self.add_knowledge(pregunta, "[REVISAR] " + respuesta_chatbot, 
                                 self._classify_question(pregunta), 0.1, "aprendizaje_negativo")
        except Exception as e:
            logging.error(f"Error en aprendizaje: {e}")

    def _classify_question(self, pregunta):
        """Clasifica pregunta automáticamente."""
        pregunta_lower = pregunta.lower()
        keywords = {
            'becas': ['beca', 'becas'],
            'ayudas_economicas': ['ayuda económica', 'ayudas económicas', 'financiamiento'],
            'carreras': ['carrera', 'carreras', 'estudios'],
            'tramites': ['matrícula', 'inscripción', 'trámite'],
            'horarios': ['horario', 'calendario', 'fecha']
        }
        
        for categoria, words in keywords.items():
            if any(word in pregunta_lower for word in words):
                return categoria
        return 'general'

    def save_to_file(self):
        """Guarda respaldo en JSON."""
        try:
            knowledge_for_json = []
            for item in self.knowledge_base:
                item_copy = item.copy()
                for key in ['fecha_creacion']:
                    if key in item_copy:
                        item_copy[key] = str(item_copy[key])
                knowledge_for_json.append(item_copy)
            
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_for_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error guardando archivo: {e}")

    def load_from_file(self):
        """Carga desde archivo como respaldo."""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                
                if self.knowledge_base:
                    questions = [item['pregunta'].lower() for item in self.knowledge_base]
                    self.vectorizer = TfidfVectorizer(stop_words='spanish')
                    self.question_vectors = self.vectorizer.fit_transform(questions)
        except Exception as e:
            logging.error(f"Error cargando archivo: {e}")
            self.knowledge_base = []

    def get_knowledge_stats(self):
        """Obtiene estadísticas básicas."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN validado = TRUE THEN 1 ELSE 0 END) as validados,
                       AVG(confianza) as confianza_promedio
                FROM knowledge_base
            """)
            stats = cursor.fetchone()
            cursor.close()
            return {'general': stats}
        except Exception as e:
            logging.error(f"Error obteniendo estadísticas: {e}")
            return {}

# Instancia global
knowledge_manager = KnowledgeManager()
