#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.knowledge_manager import KnowledgeManager
from services.chatbot import ChatbotService

def main_menu():
    km = KnowledgeManager()
    
    while True:
        print("\n" + "="*40)
        print("ADMIN CONOCIMIENTO - POCHIBOT")
        print("="*40)
        print("1. Añadir conocimiento")
        print("2. Buscar conocimiento")
        print("3. Ver estadísticas")
        print("4. Probar chatbot")
        print("5. Añadir ejemplos de becas")
        print("0. Salir")
        print("-"*40)
        
        choice = input("Opción: ").strip()
        
        if choice == "1":
            add_knowledge(km)
        elif choice == "2":
            search_knowledge(km)
        elif choice == "3":
            show_stats(km)
        elif choice == "4":
            test_chatbot()
        elif choice == "5":
            add_examples(km)
        elif choice == "0":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida.")

def add_knowledge(km):
    print("\n--- AÑADIR CONOCIMIENTO ---")
    pregunta = input("Pregunta: ").strip()
    respuesta = input("Respuesta: ").strip()
    
    if not pregunta or not respuesta:
        print("❌ Pregunta y respuesta son obligatorias")
        return
    
    categorias = {"1": "becas", "2": "ayudas_economicas", "3": "carreras", "4": "general"}
    print("Categorías: 1=becas, 2=ayudas, 3=carreras, 4=general")
    cat = categorias.get(input("Categoría (1-4): "), "general")
    
    if km.add_knowledge(pregunta, respuesta, cat, 1.0, "admin"):
        print("✅ Conocimiento añadido")
    else:
        print("❌ Error o pregunta similar existe")

def search_knowledge(km):
    print("\n--- BUSCAR CONOCIMIENTO ---")
    query = input("Búsqueda: ").strip()
    
    if not query:
        print("❌ Búsqueda vacía")
        return
    
    result = km.get_answer(query, 0.5)
    if result:
        print(f"✅ Encontrado (similitud: {result['similarity']:.2f})")
        print(f"Respuesta: {result['respuesta']}")
    else:
        print("❌ No encontrado")

def show_stats(km):
    print("\n--- ESTADÍSTICAS ---")
    stats = km.get_knowledge_stats()
    if stats and 'general' in stats:
        g = stats['general']
        print(f"Total: {g.get('total', 0)}")
        print(f"Validados: {g.get('validados', 0)}")
        print(f"Confianza promedio: {g.get('confianza_promedio', 0):.2f}")
    else:
        print("❌ No hay estadísticas")

def test_chatbot():
    print("\n--- PROBAR CHATBOT ---")
    print("Escribe 'salir' para volver")
    
    ChatbotService.initialize()
    while True:
        user_input = input("\nTú: ").strip()
        if user_input.lower() == 'salir':
            break
        if user_input:
            response = ChatbotService.get_response(user_input)
            print(f"PochiBot: {response}")

def add_examples(km):
    print("\n--- AÑADIR EJEMPLOS ---")
    ejemplos = [
        ("¿Cuáles son los requisitos para becas?", 
         "Requisitos para becas:\n• Promedio mínimo 8.5\n• Sin materias perdidas\n• Certificado de notas\n• Carta de motivación", 
         "becas"),
        ("¿Qué carreras hay en ESPOCH Orellana?", 
         "Carreras disponibles:\n• Agronomía\n• Turismo\n• Ingeniería Ambiental\n• Zootecnia\n• Tecnologías de la Información", 
         "carreras"),
        ("¿Cuándo abren convocatorias de becas?", 
         "Convocatorias:\n• Primer semestre: Marzo-Abril\n• Segundo semestre: Agosto-Septiembre", 
         "becas")
    ]
    
    added = 0
    for pregunta, respuesta, categoria in ejemplos:
        if km.add_knowledge(pregunta, respuesta, categoria, 1.0, "ejemplos"):
            added += 1
    
    print(f"✅ Añadidos {added} ejemplos")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n¡Hasta luego!")
    except Exception as e:
        print(f"\nError inesperado: {e}")
