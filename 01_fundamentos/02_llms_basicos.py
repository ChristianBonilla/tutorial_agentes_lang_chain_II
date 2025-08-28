"""
🤖 MÓDULO 2: LLMs BÁSICOS
=========================

En este módulo profundizarás en el uso de Large Language Models con LangChain.
Aprenderás sobre diferentes modelos, configuraciones avanzadas y mejores prácticas.

🎯 OBJETIVOS DE APRENDIZAJE:
- Entender los diferentes tipos de LLMs disponibles
- Configurar parámetros avanzados de LLMs
- Manejar errores y excepciones
- Optimizar el rendimiento y costos
- Trabajar con diferentes proveedores de LLMs

📚 CONCEPTOS CLAVE:
- Chat Models vs Completion Models
- Parámetros de configuración (temperature, max_tokens, etc.)
- Manejo de errores y rate limiting
- Costos y optimización
- Modelos de diferentes proveedores

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os
import time
from typing import List, Dict, Any

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config, print_config_status
from utils.helpers import print_separator, print_step, print_result, measure_execution_time

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import get_openai_callback
import openai

def tipos_llms():
    """
    🔍 Diferentes tipos de LLMs en LangChain
    """
    print_separator("TIPOS DE LLMs")
    
    print("""
    🤖 LangChain soporta diferentes tipos de LLMs:
    
    1. 📝 Chat Models (Modelos de Chat):
       - Diseñados para conversaciones
       - Aceptan mensajes estructurados
       - Ejemplos: GPT-3.5-turbo, GPT-4, Claude
       - Mejor para aplicaciones conversacionales
    
    2. 🔄 Completion Models (Modelos de Completación):
       - Diseñados para completar texto
       - Aceptan texto simple
       - Ejemplos: text-davinci-003, text-curie-001
       - Mejor para generación de texto
    
    3. 🌐 Diferentes Proveedores:
       - OpenAI (GPT-3.5, GPT-4)
       - Anthropic (Claude)
       - Google (PaLM, Gemini)
       - Hugging Face (modelos locales)
       - Cohere
    """)

def chat_models_basicos():
    """
    💬 Trabajando con Chat Models
    """
    print_step(1, "Chat Models Básicos", "Creando y configurando modelos de chat")
    
    try:
        # Modelo básico
        chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=150
        )
        
        print_result("Chat Model Básico", chat_model)
        
        # Mensajes estructurados
        mensajes = [
            SystemMessage(content="Eres un asistente experto en programación Python."),
            HumanMessage(content="¿Qué es una función lambda?")
        ]
        
        print(f"\n📤 Enviando mensajes estructurados...")
        respuesta = chat_model.invoke(mensajes)
        
        print(f"📥 Respuesta: {respuesta.content}")
        
        return chat_model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def completion_models():
    """
    🔄 Trabajando con Completion Models
    """
    print_step(2, "Completion Models", "Usando modelos de completación")
    
    try:
        # Modelo de completación
        completion_model = OpenAI(
            model="text-davinci-003",
            temperature=0.5,
            max_tokens=100
        )
        
        print_result("Completion Model", completion_model)
        
        # Texto simple
        prompt = "Explica qué es una API REST en una frase:"
        
        print(f"\n📤 Enviando prompt: {prompt}")
        respuesta = completion_model.invoke(prompt)
        
        print(f"📥 Respuesta: {respuesta}")
        
        return completion_model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def parametros_avanzados():
    """
    ⚙️ Configuración avanzada de parámetros
    """
    print_step(3, "Parámetros Avanzados", "Explorando opciones de configuración")
    
    parametros = {
        "temperature": {
            "descripcion": "Controla la aleatoriedad (0-1)",
            "valores": [0.1, 0.5, 0.9],
            "efecto": "0=determinístico, 1=muy creativo"
        },
        "max_tokens": {
            "descripcion": "Máximo número de tokens en respuesta",
            "valores": [50, 100, 200],
            "efecto": "Controla la longitud de la respuesta"
        },
        "top_p": {
            "descripcion": "Núcleo de probabilidad (0-1)",
            "valores": [0.1, 0.5, 0.9],
            "efecto": "Controla la diversidad del vocabulario"
        },
        "frequency_penalty": {
            "descripcion": "Penalización por frecuencia (-2 a 2)",
            "valores": [-1, 0, 1],
            "efecto": "Reduce repetición de palabras"
        },
        "presence_penalty": {
            "descripcion": "Penalización por presencia (-2 a 2)",
            "valores": [-1, 0, 1],
            "efecto": "Reduce repetición de temas"
        }
    }
    
    print("\n⚙️ PARÁMETROS DE CONFIGURACIÓN:")
    for param, info in parametros.items():
        print(f"\n🔹 {param.upper()}:")
        print(f"   Descripción: {info['descripcion']}")
        print(f"   Valores típicos: {info['valores']}")
        print(f"   Efecto: {info['efecto']}")

def experimentar_temperature():
    """
    🌡️ Experimentando con diferentes temperaturas
    """
    print_step(4, "Experimento: Temperature", "Viendo cómo afecta la creatividad")
    
    try:
        prompt = "Escribe una historia corta sobre un robot que aprende a programar."
        
        temperaturas = [0.1, 0.5, 0.9]
        
        for temp in temperaturas:
            print(f"\n🌡️ Temperature: {temp}")
            print("-" * 30)
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=temp,
                max_tokens=100
            )
            
            respuesta = llm.invoke([HumanMessage(content=prompt)])
            print(f"Respuesta: {respuesta.content[:150]}...")
            
            time.sleep(1)  # Pausa para evitar rate limiting
            
    except Exception as e:
        print(f"❌ Error en experimento: {e}")

def manejo_errores():
    """
    🛡️ Manejo de errores y excepciones
    """
    print_step(5, "Manejo de Errores", "Aprendiendo a manejar problemas comunes")
    
    errores_comunes = {
        "Rate Limiting": "Demasiadas solicitudes en poco tiempo",
        "API Key Inválida": "Clave de API incorrecta o expirada",
        "Modelo No Disponible": "Modelo no existe o no está disponible",
        "Tokens Excedidos": "Respuesta más larga que max_tokens",
        "Timeout": "La solicitud tardó demasiado"
    }
    
    print("\n🛡️ ERRORES COMUNES Y SOLUCIONES:")
    for error, descripcion in errores_comunes.items():
        print(f"   🔹 {error}: {descripcion}")
    
    # Ejemplo de manejo de errores
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50,
            request_timeout=10  # Timeout de 10 segundos
        )
        
        respuesta = llm.invoke([HumanMessage(content="Hola")])
        print(f"\n✅ Solicitud exitosa: {respuesta.content}")
        
    except openai.RateLimitError:
        print("❌ Rate limit alcanzado. Espera un momento.")
    except openai.AuthenticationError:
        print("❌ Error de autenticación. Verifica tu API key.")
    except openai.APITimeoutError:
        print("❌ Timeout. La solicitud tardó demasiado.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def optimizacion_costos():
    """
    💰 Optimización de costos
    """
    print_step(6, "Optimización de Costos", "Maximizando eficiencia y minimizando gastos")
    
    print("""
    💰 ESTRATEGIAS DE OPTIMIZACIÓN:
    
    1. 📏 Control de Longitud:
       - Usa max_tokens apropiado
       - Limita el contexto de entrada
       - Evita prompts innecesariamente largos
    
    2. 🎯 Modelos Eficientes:
       - GPT-3.5-turbo es más barato que GPT-4
       - Usa modelos más pequeños para tareas simples
       - Considera modelos locales para desarrollo
    
    3. 🔄 Caching:
       - Guarda respuestas frecuentes
       - Reutiliza resultados similares
       - Implementa cache local
    
    4. 📊 Monitoreo:
       - Usa callbacks para tracking
       - Monitorea uso de tokens
       - Establece límites de gasto
    """)
    
    # Ejemplo con callback para monitoreo
    try:
        with get_openai_callback() as cb:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            respuesta = llm.invoke([HumanMessage(content="Explica qué es Python en una frase.")])
            
            print(f"\n📊 ESTADÍSTICAS DE USO:")
            print(f"   Tokens totales: {cb.total_tokens}")
            print(f"   Tokens de prompt: {cb.prompt_tokens}")
            print(f"   Tokens de respuesta: {cb.completion_tokens}")
            print(f"   Costo total: ${cb.total_cost}")
            
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

def diferentes_proveedores():
    """
    🌐 Trabajando con diferentes proveedores
    """
    print_step(7, "Diferentes Proveedores", "Explorando opciones de LLMs")
    
    proveedores = {
        "OpenAI": {
            "modelos": ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"],
            "ventajas": "Muy estable, buena documentación",
            "desventajas": "Puede ser costoso"
        },
        "Anthropic": {
            "modelos": ["claude-2", "claude-instant"],
            "ventajas": "Muy bueno para análisis",
            "desventajas": "API más nueva"
        },
        "Google": {
            "modelos": ["text-bison", "chat-bison"],
            "ventajas": "Integración con Google Cloud",
            "desventajas": "Menos documentación"
        },
        "Hugging Face": {
            "modelos": ["miles de modelos locales"],
            "ventajas": "Gratis, privado",
            "desventajas": "Requiere más recursos"
        }
    }
    
    print("\n🌐 PROVEEDORES DE LLMs:")
    for proveedor, info in proveedores.items():
        print(f"\n🔹 {proveedor}:")
        print(f"   Modelos: {', '.join(info['modelos'])}")
        print(f"   Ventajas: {info['ventajas']}")
        print(f"   Desventajas: {info['desventajas']}")

def ejercicios_avanzados():
    """
    🎯 Ejercicios avanzados para práctica
    """
    print_step(8, "Ejercicios Avanzados", "Pon en práctica lo aprendido")
    
    ejercicios = [
        {
            "titulo": "Comparador de Modelos",
            "descripcion": "Crea una función que compare respuestas de diferentes modelos",
            "objetivo": "Entender diferencias entre modelos"
        },
        {
            "titulo": "Optimizador de Prompts",
            "descripcion": "Crea prompts que minimicen tokens pero maximicen calidad",
            "objetivo": "Aprender optimización de costos"
        },
        {
            "titulo": "Manejador de Errores",
            "descripcion": "Implementa un sistema robusto de manejo de errores",
            "objetivo": "Crear aplicaciones robustas"
        },
        {
            "titulo": "Monitor de Costos",
            "descripcion": "Crea un sistema que monitoree y limite gastos",
            "objetivo": "Control de presupuesto"
        }
    ]
    
    print("\n🎯 EJERCICIOS AVANZADOS:")
    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{i}. {ejercicio['titulo']}")
        print(f"   Descripción: {ejercicio['descripcion']}")
        print(f"   Objetivo: {ejercicio['objetivo']}")

def mejores_practicas():
    """
    ⭐ Mejores prácticas para LLMs
    """
    print_step(9, "Mejores Prácticas", "Consejos para usar LLMs efectivamente")
    
    practicas = [
        "🎯 Siempre especifica el contexto y rol del modelo",
        "📏 Usa max_tokens apropiado para tu caso de uso",
        "🌡️ Ajusta temperature según la tarea (baja para hechos, alta para creatividad)",
        "🛡️ Implementa manejo de errores robusto",
        "💰 Monitorea costos y optimiza cuando sea posible",
        "🔄 Usa caching para respuestas frecuentes",
        "📊 Implementa logging para debugging",
        "⚡ Considera modelos más pequeños para tareas simples",
        "🔒 Maneja API keys de forma segura",
        "📚 Documenta tus prompts y configuraciones"
    ]
    
    print("\n⭐ MEJORES PRÁCTICAS:")
    for practica in practicas:
        print(f"   {practica}")

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 2: LLMs BÁSICOS")
    
    # Verificar configuración
    if not config.validate_config():
        return
    
    # Contenido del módulo
    tipos_llms()
    
    # Ejemplos prácticos
    chat_models_basicos()
    completion_models()
    
    # Configuración avanzada
    parametros_avanzados()
    experimentar_temperature()
    
    # Manejo de errores y optimización
    manejo_errores()
    optimizacion_costos()
    
    # Información adicional
    diferentes_proveedores()
    mejores_practicas()
    ejercicios_avanzados()
    
    print("\n🎉 ¡Módulo 2 completado! Ahora conoces los fundamentos de LLMs.")
    print("🚀 Próximo módulo: Prompts y Templates Avanzados")

if __name__ == "__main__":
    main()
