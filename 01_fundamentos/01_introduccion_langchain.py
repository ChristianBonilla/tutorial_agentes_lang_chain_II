"""
🚀 MÓDULO 1: INTRODUCCIÓN A LANGCHAIN
====================================

Este es tu primer paso en el viaje hacia el dominio de LangChain.
Aquí aprenderás los conceptos fundamentales y verás tu primera aplicación funcionando.

🎯 OBJETIVOS DE APRENDIZAJE:
- Entender qué es LangChain y por qué es importante
- Conocer la arquitectura básica de LangChain
- Crear tu primera aplicación con LangChain
- Familiarizarte con los componentes principales

📚 CONCEPTOS CLAVE:
- LLMs (Large Language Models)
- Chains (Cadenas de procesamiento)
- Prompts (Instrucciones para LLMs)
- Components (Componentes reutilizables)

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config, print_config_status
from utils.helpers import print_separator, print_step, print_result, print_learning_tips

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

def introduccion_langchain():
    """
    🎓 Introducción conceptual a LangChain
    """
    print_separator("¿QUÉ ES LANGCHAIN?")
    
    print("""
    🤖 LangChain es una biblioteca de Python que facilita el desarrollo de 
    aplicaciones con Large Language Models (LLMs). Es como un "Lego" para 
    construir aplicaciones de IA conversacional.
    
    🎯 ¿Por qué LangChain?
    - Simplifica la integración con diferentes LLMs
    - Proporciona componentes reutilizables
    - Permite crear flujos complejos de procesamiento
    - Facilita el manejo de memoria y contexto
    - Ofrece herramientas para RAG (Retrieval Augmented Generation)
    
    🏗️ Arquitectura básica:
    - LLMs: Los modelos de lenguaje (GPT, Claude, etc.)
    - Prompts: Instrucciones que le damos a los LLMs
    - Chains: Secuencias de operaciones
    - Memory: Almacenamiento de contexto
    - Tools: Herramientas externas
    """)

def configuracion_inicial():
    """
    ⚙️ Configuración inicial del proyecto
    """
    print_step(1, "Configuración Inicial", "Verificando que todo esté listo")
    
    # Verificar configuración
    print_config_status()
    
    if not config.validate_config():
        print("""
        ⚠️  IMPORTANTE: Para continuar necesitas:
        1. Crear un archivo .env basado en env_example.txt
        2. Agregar tu API key de OpenAI
        3. Ejecutar: pip install -r requirements.txt
        """)
        return False
    
    print("✅ Configuración completada exitosamente")
    return True

def primer_llm():
    """
    🤖 Tu primer LLM con LangChain
    """
    print_step(2, "Tu Primer LLM", "Creando y usando un modelo de lenguaje")
    
    try:
        # Crear una instancia del LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=config.default_temperature,
            max_tokens=config.default_max_tokens
        )
        
        print_result("LLM Creado", llm, show_type=True)
        
        # Mensaje simple
        mensaje = HumanMessage(content="¡Hola! ¿Puedes explicarme qué es LangChain en una frase?")
        
        print(f"\n📤 Enviando mensaje: {mensaje.content}")
        
        # Obtener respuesta
        respuesta = llm.invoke([mensaje])
        
        print_result("Respuesta del LLM", respuesta)
        print(f"📥 Respuesta: {respuesta.content}")
        
        return llm
        
    except Exception as e:
        print(f"❌ Error al crear LLM: {e}")
        print("💡 Asegúrate de tener configurada tu API key de OpenAI")
        return None

def prompts_basicos():
    """
    📝 Introducción a los Prompts
    """
    print_step(3, "Prompts Básicos", "Aprendiendo a crear instrucciones efectivas")
    
    # Crear un prompt template
    template = PromptTemplate(
        input_variables=["tema", "nivel"],
        template="""
        Eres un experto instructor de programación. 
        Explica el tema '{tema}' a un nivel '{nivel}'.
        Mantén la explicación clara y concisa.
        """
    )
    
    print_result("Prompt Template", template)
    
    # Formatear el prompt
    prompt_formateado = template.format(
        tema="variables en Python",
        nivel="principiante"
    )
    
    print(f"\n📝 Prompt formateado:")
    print("-" * 40)
    print(prompt_formateado)
    print("-" * 40)
    
    return template

def primera_chain():
    """
    🔗 Tu primera Chain
    """
    print_step(4, "Tu Primera Chain", "Combinando LLM con Prompt")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Crear prompt
        prompt = PromptTemplate(
            input_variables=["concepto"],
            template="Explica qué es {concepto} en el contexto de la programación. Máximo 2 frases."
        )
        
        # Crear chain (combinación de prompt + LLM)
        from langchain.chains import LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print_result("Chain Creada", chain)
        
        # Usar la chain
        conceptos = ["API", "JSON", "REST"]
        
        for concepto in conceptos:
            print(f"\n🔍 Explicando: {concepto}")
            respuesta = chain.run(concepto)
            print(f"📝 Respuesta: {respuesta}")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error al crear chain: {e}")
        return None

def ejercicios_practicos():
    """
    🎯 Ejercicios prácticos para consolidar el aprendizaje
    """
    print_step(5, "Ejercicios Prácticos", "Pon en práctica lo aprendido")
    
    print("""
    🎯 EJERCICIOS:
    
    1. 🔧 Modifica el prompt template para que explique conceptos de IA
    2. 🌡️ Experimenta con diferentes valores de temperature (0.1, 0.5, 1.0)
    3. 📏 Cambia max_tokens y observa cómo afecta la respuesta
    4. 🎨 Crea un prompt que genere poemas sobre programación
    5. 🔄 Crea una chain que traduzca texto al inglés
    
    💡 CONSEJOS:
    - Ejecuta cada ejercicio paso a paso
    - Observa las diferencias en las respuestas
    - Experimenta con diferentes modelos si tienes acceso
    """)

def conceptos_importantes():
    """
    ⚠️ Conceptos importantes a recordar
    """
    print_step(6, "Conceptos Importantes", "Puntos clave para recordar")
    
    conceptos = {
        "LLM": "Large Language Model - El modelo de IA que genera texto",
        "Prompt": "Instrucción que le damos al LLM para guiar su respuesta",
        "Chain": "Secuencia de operaciones que combinan diferentes componentes",
        "Temperature": "Controla la creatividad/aleatoriedad de las respuestas (0-1)",
        "Max Tokens": "Límite máximo de palabras en la respuesta",
        "Template": "Plantilla de prompt con variables que se pueden reemplazar"
    }
    
    print("\n📚 CONCEPTOS CLAVE:")
    for concepto, definicion in conceptos.items():
        print(f"   🔹 {concepto}: {definicion}")

def siguiente_paso():
    """
    🚀 Preparación para el siguiente módulo
    """
    print_step(7, "Siguiente Paso", "Preparándote para LLMs Avanzados")
    
    print("""
    🎉 ¡Felicitaciones! Has completado tu primera lección de LangChain.
    
    📋 LO QUE APRENDISTE:
    ✅ Configurar un proyecto de LangChain
    ✅ Crear y usar LLMs
    ✅ Trabajar con prompts y templates
    ✅ Crear tu primera chain
    ✅ Entender los conceptos fundamentales
    
    🚀 PRÓXIMO MÓDULO: LLMs Básicos
    - Diferentes tipos de LLMs
    - Configuración avanzada
    - Manejo de errores
    - Optimización de respuestas
    
    💡 RECUERDA: La práctica es la clave del aprendizaje
    """)

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 1: INTRODUCCIÓN A LANGCHAIN")
    
    # Verificar configuración
    if not configuracion_inicial():
        return
    
    # Contenido del módulo
    introduccion_langchain()
    
    # Ejemplos prácticos
    llm = primer_llm()
    if llm:
        prompts_basicos()
        primera_chain()
    
    # Consolidación
    conceptos_importantes()
    ejercicios_practicos()
    siguiente_paso()
    
    # Consejos de aprendizaje
    print_learning_tips()

if __name__ == "__main__":
    main()
