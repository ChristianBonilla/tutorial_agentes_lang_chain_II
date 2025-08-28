"""
🛠️ Utilidades y Funciones Auxiliares para LangChain
==================================================

Este módulo contiene funciones auxiliares que facilitan el trabajo con LangChain
y proporcionan utilidades comunes para todos los ejemplos.

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import time
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_separator(title: str, char: str = "=", length: int = 60):
    """
    📏 Imprime un separador con título para organizar la salida
    
    Args:
        title: Título del separador
        char: Carácter para el separador
        length: Longitud del separador
    """
    print(f"\n{char * length}")
    print(f"{title.center(length)}")
    print(f"{char * length}\n")

def print_step(step_number: int, title: str, description: str = ""):
    """
    📋 Imprime un paso del tutorial de manera organizada
    
    Args:
        step_number: Número del paso
        title: Título del paso
        description: Descripción opcional
    """
    print(f"\n🔹 PASO {step_number}: {title}")
    if description:
        print(f"   {description}")
    print("-" * 50)

def print_result(title: str, result: Any, show_type: bool = True):
    """
    📊 Imprime un resultado de manera formateada
    
    Args:
        title: Título del resultado
        result: Resultado a mostrar
        show_type: Si mostrar el tipo de dato
    """
    print(f"\n📊 {title}:")
    print("-" * 30)
    if show_type:
        print(f"Tipo: {type(result).__name__}")
    print(f"Resultado: {result}")
    if hasattr(result, '__dict__'):
        print(f"Atributos: {list(result.__dict__.keys())}")

def measure_execution_time(func):
    """
    ⏱️ Decorador para medir el tiempo de ejecución de una función
    
    Args:
        func: Función a decorar
        
    Returns:
        Función decorada
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️  Tiempo de ejecución: {execution_time:.2f} segundos")
        return result
    return wrapper

def save_to_json(data: Any, filename: str, indent: int = 2):
    """
    💾 Guarda datos en formato JSON
    
    Args:
        data: Datos a guardar
        filename: Nombre del archivo
        indent: Indentación del JSON
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        print(f"✅ Datos guardados en {filename}")
    except Exception as e:
        print(f"❌ Error al guardar {filename}: {e}")

def load_from_json(filename: str) -> Any:
    """
    📂 Carga datos desde un archivo JSON
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        Datos cargados
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Datos cargados desde {filename}")
        return data
    except Exception as e:
        print(f"❌ Error al cargar {filename}: {e}")
        return None

def create_sample_texts() -> List[str]:
    """
    📝 Crea textos de ejemplo para usar en los ejercicios
    
    Returns:
        Lista de textos de ejemplo
    """
    return [
        "LangChain es una biblioteca para desarrollar aplicaciones con LLMs. "
        "Proporciona herramientas para conectar LLMs con fuentes de datos externas "
        "y permitir interacciones con su entorno.",
        
        "Los LLMs (Large Language Models) son modelos de lenguaje entrenados "
        "con grandes cantidades de texto. Pueden generar texto, responder preguntas "
        "y realizar tareas de procesamiento de lenguaje natural.",
        
        "El RAG (Retrieval Augmented Generation) es una técnica que combina "
        "la recuperación de información con la generación de texto. Permite "
        "que los LLMs accedan a información actualizada y específica."
    ]

def create_sample_questions() -> List[str]:
    """
    ❓ Crea preguntas de ejemplo para usar en los ejercicios
    
    Returns:
        Lista de preguntas de ejemplo
    """
    return [
        "¿Qué es LangChain y para qué se usa?",
        "¿Cuáles son las ventajas de usar LLMs?",
        "¿Cómo funciona el RAG?",
        "¿Qué son las chains en LangChain?",
        "¿Cuál es la diferencia entre un agente y una chain?"
    ]

def format_chain_output(output: Any) -> str:
    """
    🎨 Formatea la salida de una chain para mejor legibilidad
    
    Args:
        output: Salida de la chain
        
    Returns:
        Salida formateada
    """
    if isinstance(output, str):
        return output
    
    if hasattr(output, 'content'):
        return str(output.content)
    
    if hasattr(output, 'text'):
        return str(output.text)
    
    return str(output)

def validate_llm_response(response: Any, expected_keys: Optional[List[str]] = None) -> bool:
    """
    ✅ Valida que una respuesta de LLM tenga la estructura esperada
    
    Args:
        response: Respuesta del LLM
        expected_keys: Claves esperadas en la respuesta
        
    Returns:
        True si la respuesta es válida
    """
    if not response:
        print("❌ Respuesta vacía")
        return False
    
    if expected_keys:
        if isinstance(response, dict):
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                print(f"❌ Faltan claves esperadas: {missing_keys}")
                return False
    
    print("✅ Respuesta válida")
    return True

def create_exercise_template(exercise_name: str, description: str) -> str:
    """
    📋 Crea una plantilla para ejercicios
    
    Args:
        exercise_name: Nombre del ejercicio
        description: Descripción del ejercicio
        
    Returns:
        Plantilla del ejercicio
    """
    template = f"""
# 🎯 EJERCICIO: {exercise_name}
# ================================

# Descripción: {description}

# Tu código aquí:
# TODO: Implementa la solución

# Verificación:
# TODO: Agrega verificaciones para tu solución

print("✅ Ejercicio completado!")
"""
    return template

def print_learning_tips():
    """
    💡 Imprime consejos de aprendizaje
    """
    tips = [
        "🔍 Lee cada línea de código y entiende qué hace",
        "🧪 Experimenta modificando los parámetros",
        "📝 Toma notas de los conceptos nuevos",
        "🔄 Ejecuta los ejemplos múltiples veces",
        "❓ Si algo no está claro, investiga en la documentación",
        "💻 Practica creando tus propios ejemplos",
        "🤝 Comparte tus dudas y descubrimientos"
    ]
    
    print("\n💡 CONSEJOS DE APRENDIZAJE:")
    print("=" * 40)
    for tip in tips:
        print(f"   {tip}")
    print("=" * 40)

def print_next_steps():
    """
    🚀 Imprime los siguientes pasos recomendados
    """
    print("\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
    print("=" * 40)
    print("1. 📚 Revisa la documentación oficial de LangChain")
    print("2. 🎯 Completa los ejercicios de cada módulo")
    print("3. 🔧 Experimenta con diferentes modelos y parámetros")
    print("4. 🏗️  Construye tu propio proyecto usando LangChain")
    print("5. 🤝 Únete a la comunidad de LangChain")
    print("=" * 40)

if __name__ == "__main__":
    # Ejemplo de uso de las utilidades
    print_separator("UTILIDADES DE LANGCHAIN")
    print_step(1, "Crear textos de ejemplo", "Generando textos para ejercicios")
    texts = create_sample_texts()
    print_result("Textos de ejemplo", texts[:1])
    
    print_learning_tips()
    print_next_steps()
