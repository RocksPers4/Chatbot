#!/usr/bin/env python3
"""
Script de administración para la base de conocimiento del chatbot ESPOCH.
Permite gestionar preguntas y respuestas, validar conocimiento y ver estadísticas.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.knowledge_manager import KnowledgeManager
from services.chatbot import ChatbotService
import json

def main_menu():
    """Muestra el menú principal de administración."""
    km = KnowledgeManager()
    
    while True:
        print("\n" + "="*50)
        print("ADMINISTRADOR DE CONOCIMIENTO - POCHIBOT")
        print("="*50)
        print("1. Añadir nuevo conocimiento")
        print("2. Buscar conocimiento existente")
        print("3. Ver conocimiento no validado")
        print("4. Validar conocimiento")
        print("5. Ver estadísticas")
        print("6. Exportar base de conocimiento")
        print("7. Importar conocimiento")
        print("8. Probar chatbot")
        print("9. Añadir ejemplos de becas")
        print("0. Salir")
        print("-"*50)
        
        choice = input("Selecciona una opción: ").strip()
        
        if choice == "1":
            add_knowledge(km)
        elif choice == "2":
            search_knowledge(km)
        elif choice == "3":
            show_unvalidated(km)
        elif choice == "4":
            validate_knowledge(km)
        elif choice == "5":
            show_stats(km)
        elif choice == "6":
            export_knowledge(km)
        elif choice == "7":
            import_knowledge(km)
        elif choice == "8":
            test_chatbot()
        elif choice == "9":
            add_scholarship_examples(km)
        elif choice == "0":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

def add_knowledge(km):
    """Añade nuevo conocimiento a la base de datos."""
    print("\n--- AÑADIR NUEVO CONOCIMIENTO ---")
    
    pregunta = input("Pregunta: ").strip()
    if not pregunta:
        print("La pregunta no puede estar vacía.")
        return
    
    respuesta = input("Respuesta: ").strip()
    if not respuesta:
        print("La respuesta no puede estar vacía.")
        return
    
    print("\nCategorías disponibles:")
    print("1. becas")
    print("2. ayudas_economicas")
    print("3. carreras")
    print("4. tramites")
    print("5. horarios")
    print("6. general")
    
    cat_choice = input("Selecciona categoría (1-6): ").strip()
    categorias = {
        "1": "becas",
        "2": "ayudas_economicas", 
        "3": "carreras",
        "4": "tramites",
        "5": "horarios",
        "6": "general"
    }
    categoria = categorias.get(cat_choice, "general")
    
    try:
        confianza = float(input("Confianza (0.0-1.0, default 1.0): ") or "1.0")
        confianza = max(0.0, min(1.0, confianza))
    except ValueError:
        confianza = 1.0
    
    success = km.add_knowledge(pregunta, respuesta, categoria, confianza, "admin")
    
    if success:
        print("✅ Conocimiento añadido exitosamente!")
    else:
        print("❌ No se pudo añadir. Puede que ya exista una pregunta similar.")

def search_knowledge(km):
    """Busca conocimiento existente."""
    print("\n--- BUSCAR CONOCIMIENTO ---")
    
    query = input("Ingresa tu búsqueda: ").strip()
    if not query:
        print("La búsqueda no puede estar vacía.")
        return
    
    result = km.get_answer(query, threshold=0.5)
    
    if result:
        print(f"\n✅ RESULTADO ENCONTRADO:")
        print(f"Similitud: {result['similarity']:.2f}")
        print(f"Confianza: {result['confianza']:.2f}")
        print(f"Categoría: {result['categoria']}")
        print(f"Respuesta: {result['respuesta']}")
    else:
        print("❌ No se encontraron resultados similares.")

def show_unvalidated(km):
    """Muestra conocimiento no validado."""
    print("\n--- CONOCIMIENTO NO VALIDADO ---")
    
    unvalidated = km.get_unvalidated_knowledge()
    
    if not unvalidated:
        print("✅ Todo el conocimiento está validado.")
        return
    
    for i, item in enumerate(unvalidated, 1):
        print(f"\n{i}. ID: {item['id']}")
        print(f"   Pregunta: {item['pregunta']}")
        print(f"   Respuesta: {item['respuesta'][:100]}...")
        print(f"   Categoría: {item['categoria']}")
        print(f"   Confianza: {item['confianza']}")
        print(f"   Fuente: {item['fuente']}")
        print(f"   Fecha: {item['fecha_creacion']}")

def validate_knowledge(km):
    """Valida o invalida conocimiento específico."""
    print("\n--- VALIDAR CONOCIMIENTO ---")
    
    try:
        knowledge_id = int(input("ID del conocimiento a validar: "))
        action = input("¿Validar? (s/n): ").lower().strip()
        
        if action in ['s', 'si', 'yes', 'y']:
            km.validate_knowledge(knowledge_id, True)
            print("✅ Conocimiento validado.")
        elif action in ['n', 'no']:
            km.validate_knowledge(knowledge_id, False)
            print("❌ Conocimiento marcado como no válido.")
        else:
            print("Acción no reconocida.")
            
    except ValueError:
        print("ID debe ser un número.")
    except Exception as e:
        print(f"Error: {e}")

def show_stats(km):
    """Muestra estadísticas de la base de conocimiento."""
    print("\n--- ESTADÍSTICAS DE LA BASE DE CONOCIMIENTO ---")
    
    stats = km.get_knowledge_stats()
    
    if stats:
        general = stats.get('general', {})
        print(f"📊 ESTADÍSTICAS GENERALES:")
        print(f"   Total de entradas: {general.get('total', 0)}")
        print(f"   Validadas: {general.get('validados', 0)}")
        print(f"   Activas: {general.get('activos', 0)}")
        print(f"   Confianza promedio: {general.get('confianza_promedio', 0):.2f}")
        print(f"   Categorías: {general.get('categorias', 0)}")
        
        categories = stats.get('por_categoria', [])
        if categories:
            print(f"\n📈 POR CATEGORÍA:")
            for cat in categories:
                print(f"   {cat['categoria']}: {cat['cantidad']} entradas")
    else:
        print("❌ No se pudieron obtener estadísticas.")

def export_knowledge(km):
    """Exporta la base de conocimiento."""
    print("\n--- EXPORTAR BASE DE CONOCIMIENTO ---")
    
    filename = input("Nombre del archivo (opcional): ").strip()
    
    exported_file = km.export_knowledge(filename if filename else None)
    
    if exported_file:
        print(f"✅ Base de conocimiento exportada a: {exported_file}")
    else:
        print("❌ Error al exportar.")

def import_knowledge(km):
    """Importa conocimiento desde un archivo."""
    print("\n--- IMPORTAR CONOCIMIENTO ---")
    
    filename = input("Ruta del archivo a importar: ").strip()
    
    if not os.path.exists(filename):
        print("❌ El archivo no existe.")
        return
    
    imported_count = km.import_knowledge(filename)
    
    if imported_count > 0:
        print(f"✅ Se importaron {imported_count} entradas de conocimiento.")
    else:
        print("❌ No se pudo importar conocimiento.")

def test_chatbot():
    """Prueba el chatbot interactivamente."""
    print("\n--- PROBAR CHATBOT ---")
    print("Escribe 'salir' para volver al menú principal.")
    
    ChatbotService.initialize()
    
    while True:
        user_input = input("\nTú: ").strip()
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            break
        
        if not user_input:
            continue
        
        response = ChatbotService.get_response(user_input)
        print(f"PochiBot: {response}")

def add_scholarship_examples(km):
    """Añade ejemplos predefinidos sobre becas."""
    print("\n--- AÑADIR EJEMPLOS DE BECAS ---")
    
    ejemplos_becas = [
        {
            "pregunta": "¿Cuáles son los requisitos para la beca de excelencia académica?",
            "respuesta": "Para la beca de excelencia académica necesitas:\n\n• Promedio mínimo de 8.5 puntos\n• No tener materias perdidas o reprobadas\n• Presentar certificado de notas actualizado\n• Carta de motivación explicando por qué mereces la beca\n• Certificado de matrícula vigente\n• Copia de cédula de identidad\n\nLa documentación debe presentarse en el Departamento de Bienestar Estudiantil.",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Cuándo abren las convocatorias de becas en la ESPOCH?",
            "respuesta": "Las convocatorias de becas en la ESPOCH Sede Orellana se abren:\n\n• Primer semestre: Marzo - Abril\n• Segundo semestre: Agosto - Septiembre\n\nTe recomendamos:\n• Estar atento a las publicaciones oficiales\n• Seguir las redes sociales de la ESPOCH\n• Consultar regularmente en Bienestar Estudiantil\n• Preparar tu documentación con anticipación",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Qué tipos de becas ofrece la ESPOCH Sede Orellana?",
            "respuesta": "La ESPOCH Sede Orellana ofrece varios tipos de becas:\n\n• Beca de Excelencia Académica (para estudiantes destacados)\n• Beca Socioeconómica (para estudiantes de bajos recursos)\n• Beca Deportiva (para atletas destacados)\n• Beca Cultural (para estudiantes con talentos artísticos)\n• Beca de Investigación (para proyectos de investigación)\n\nCada beca tiene requisitos específicos y diferentes porcentajes de cobertura.",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Cómo puedo aplicar a una beca en la ESPOCH?",
            "respuesta": "Para aplicar a una beca en la ESPOCH Sede Orellana:\n\n1. Revisa las convocatorias vigentes\n2. Verifica que cumples los requisitos\n3. Reúne toda la documentación necesaria\n4. Presenta tu solicitud en Bienestar Estudiantil\n5. Espera la evaluación del comité\n6. Asiste a la entrevista si es requerida\n\n¿Necesitas ayuda con algún paso específico del proceso?",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Qué ayudas económicas hay disponibles además de las becas?",
            "respuesta": "Además de las becas, la ESPOCH Sede Orellana ofrece:\n\n• Ayuda alimentaria (descuentos en el comedor)\n• Subsidio de transporte\n• Programa de trabajo estudiantil\n• Ayuda para materiales de estudio\n• Descuentos en servicios médicos\n• Apoyo para actividades extracurriculares\n\nEstas ayudas tienen requisitos socioeconómicos específicos.",
            "categoria": "ayudas_economicas"
        },
        {
            "pregunta": "¿Puedo tener más de una beca al mismo tiempo?",
            "respuesta": "En general, no es posible tener múltiples becas simultáneamente en la ESPOCH. Sin embargo:\n\n• Puedes combinar una beca con ayudas económicas menores\n• Algunas ayudas específicas (como transporte o alimentación) pueden complementar una beca principal\n• Debes consultar en Bienestar Estudiantil sobre compatibilidades específicas\n\nLa prioridad siempre se da a estudiantes que no tienen ningún tipo de apoyo económico.",
            "categoria": "becas"
        },
        {
            "pregunta": "¿Qué pasa si pierdo mi beca por bajo rendimiento?",
            "respuesta": "Si pierdes tu beca por bajo rendimiento académico:\n\n• Tienes un período de gracia de un semestre para mejorar\n• Puedes solicitar tutoría académica gratuita\n• Debes presentar un plan de mejoramiento académico\n• Puedes volver a aplicar en la siguiente convocatoria si cumples los requisitos\n• El Departamento de Bienestar Estudiantil te orientará sobre alternativas\n\nEs importante comunicarte inmediatamente con Bienestar Estudiantil si tienes dificultades académicas.",
            "categoria": "becas"
        }
    ]
    
    added_count = 0
    for ejemplo in ejemplos_becas:
        if km.add_knowledge(**ejemplo, confianza=1.0, fuente="ejemplos_admin"):
            added_count += 1
    
    print(f"✅ Se añadieron {added_count} ejemplos de becas a la base de conocimiento.")
    
    if added_count < len(ejemplos_becas):
        print(f"⚠️  {len(ejemplos_becas) - added_count} ejemplos ya existían o no se pudieron añadir.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
    except Exception as e:
        print(f"\nError inesperado: {e}")
