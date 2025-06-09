#!/usr/bin/env python3
"""
Selector de modelos para el chatbot ESPOCH.
Permite cambiar entre diferentes modelos de Hugging Face.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.chatbot import ChatbotService

def main():
    print("🤖 SELECTOR DE MODELOS - POCHIBOT")
    print("="*50)
    
    # Mostrar modelos disponibles
    print("\nModelos disponibles:")
    for key, config in ChatbotService.AVAILABLE_MODELS.items():
        print(f"\n{key}:")
        print(f"  📦 Modelo: {config['name']}")
        print(f"  🔧 Tipo: {config['type']}")
        print(f"  📝 Descripción: {config['description']}")
    
    print(f"\n🎯 Modelo actual: {ChatbotService.CURRENT_MODEL}")
    
    while True:
        print("\n" + "-"*50)
        print("1. Cambiar modelo")
        print("2. Probar modelo actual")
        print("3. Ver información del modelo")
        print("4. Comparar modelos")
        print("0. Salir")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == "1":
            change_model()
        elif choice == "2":
            test_current_model()
        elif choice == "3":
            show_model_info()
        elif choice == "4":
            compare_models()
        elif choice == "0":
            print("¡Hasta luego!")
            break
        else:
            print("Opción no válida.")

def change_model():
    """Cambia el modelo del chatbot."""
    print("\n--- CAMBIAR MODELO ---")
    
    print("Modelos disponibles:")
    models = list(ChatbotService.AVAILABLE_MODELS.keys())
    for i, model in enumerate(models, 1):
        config = ChatbotService.AVAILABLE_MODELS[model]
        print(f"{i}. {model} - {config['name']}")
    
    try:
        choice = int(input(f"\nSelecciona modelo (1-{len(models)}): "))
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            print(f"\n🔄 Cambiando a {selected_model}...")
            
            result = ChatbotService.change_model(selected_model)
            print(f"✅ {result}")
        else:
            print("❌ Opción no válida.")
    except ValueError:
        print("❌ Debe ser un número.")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_current_model():
    """Prueba el modelo actual."""
    print("\n--- PROBAR MODELO ACTUAL ---")
    
    info = ChatbotService.get_model_info()
    print(f"🤖 Modelo: {info['nombre']}")
    print(f"📝 Descripción: {info['descripcion']}")
    print("\nEscribe 'salir' para volver al menú.")
    
    try:
        ChatbotService.initialize()
        
        # Preguntas de prueba sugeridas
        test_questions = [
            "¿Qué es la ESPOCH?",
            "¿Cuáles son los requisitos para becas?",
            "¿Qué carreras hay disponibles?",
            "Hola, ¿cómo estás?"
        ]
        
        print("\n💡 Preguntas de prueba sugeridas:")
        for i, q in enumerate(test_questions, 1):
            print(f"{i}. {q}")
        
        while True:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not user_input:
                continue
            
            # Verificar si es un número de pregunta sugerida
            try:
                if user_input.isdigit():
                    num = int(user_input)
                    if 1 <= num <= len(test_questions):
                        user_input = test_questions[num - 1]
                        print(f"Pregunta seleccionada: {user_input}")
            except:
                pass
            
            response = ChatbotService.get_response(user_input)
            print(f"PochiBot: {response}")
            
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")

def show_model_info():
    """Muestra información detallada del modelo actual."""
    print("\n--- INFORMACIÓN DEL MODELO ---")
    
    info = ChatbotService.get_model_info()
    
    print(f"🎯 Modelo actual: {info['modelo_actual']}")
    print(f"📦 Nombre completo: {info['nombre']}")
    print(f"🔧 Tipo: {info['tipo']}")
    print(f"📝 Descripción: {info['descripcion']}")
    
    # Información adicional según el tipo
    if info['tipo'] == 'seq2seq':
        print("\n📋 Características de modelos Seq2Seq:")
        print("• Excelentes para tareas de generación de texto")
        print("• Pueden seguir instrucciones específicas")
        print("• Buenos para resumir y responder preguntas")
        print("• Requieren prompts bien estructurados")
    elif info['tipo'] == 'causal':
        print("\n📋 Características de modelos Causales:")
        print("• Especializados en conversaciones naturales")
        print("• Generan texto de manera fluida")
        print("• Buenos para diálogos interactivos")
        print("• Pueden mantener contexto conversacional")

def compare_models():
    """Compara diferentes modelos disponibles."""
    print("\n--- COMPARACIÓN DE MODELOS ---")
    
    print("📊 Comparación por características:\n")
    
    print("🎯 MEJOR PARA SEGUIR INSTRUCCIONES:")
    print("   1. flan_t5 - Excelente para tareas específicas")
    print("   2. mt5 - Muy bueno multilingüe")
    print("   3. mbart - Bueno para generación estructurada")
    
    print("\n💬 MEJOR PARA CONVERSACIONES NATURALES:")
    print("   1. blenderbot - Especializado en diálogos")
    print("   2. spanish_gpt2 - Muy bueno en español")
    print("   3. flan_t5 - Bueno siguiendo contexto")
    
    print("\n🌍 MEJOR SOPORTE MULTILINGÜE:")
    print("   1. mbart - Excelente multilingüe")
    print("   2. mt5 - Muy bueno multilingüe")
    print("   3. flan_t5 - Bueno en varios idiomas")
    
    print("\n🇪🇸 MEJOR PARA ESPAÑOL:")
    print("   1. spanish_gpt2 - Entrenado específicamente en español")
    print("   2. mbart - Muy bueno multilingüe")
    print("   3. mt5 - Bueno en español")
    
    print("\n⚡ VELOCIDAD Y EFICIENCIA:")
    print("   1. flan_t5 - Rápido y eficiente")
    print("   2. spanish_gpt2 - Moderadamente rápido")
    print("   3. blenderbot - Rápido para conversación")
    
    print("\n💡 RECOMENDACIONES:")
    print("• Para un chatbot universitario: flan_t5 o mt5")
    print("• Para conversaciones casuales: blenderbot")
    print("• Para español específico: spanish_gpt2")
    print("• Para versatilidad: mbart")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego!")
    except Exception as e:
        print(f"\nError inesperado: {e}")
