"""
🔗 MÓDULO 4: CHAINS SIMPLES
===========================

En este módulo aprenderás a crear y usar chains (cadenas) en LangChain.
Las chains son secuencias de operaciones que combinan diferentes componentes
para crear flujos de procesamiento más complejos.

🎯 OBJETIVOS DE APRENDIZAJE:
- Entender qué son las chains y cómo funcionan
- Crear chains simples y secuenciales
- Combinar diferentes tipos de chains
- Optimizar el rendimiento de las chains
- Manejar errores en chains

📚 CONCEPTOS CLAVE:
- LLMChain
- SequentialChain
- RouterChain
- Chain Composition
- Error Handling
- Performance Optimization

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os
from typing import List, Dict, Any

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from utils.helpers import print_separator, print_step, print_result

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback

def introduccion_chains():
    """
    🎓 Introducción a las Chains
    """
    print_separator("¿QUÉ SON LAS CHAINS?")
    
    print("""
    🔗 Las Chains son el corazón de LangChain. Son secuencias de operaciones
    que combinan diferentes componentes para crear flujos de procesamiento.
    
    🏗️ ARQUITECTURA DE CHAINS:
    
    Input → [Chain 1] → [Chain 2] → [Chain 3] → Output
    
    📋 TIPOS DE CHAINS:
    
    1. 🔄 LLMChain: La más básica, combina LLM + Prompt
    2. 🔗 SequentialChain: Ejecuta chains en secuencia
    3. 🛣️ RouterChain: Dirige el flujo según condiciones
    4. 🔀 TransformChain: Transforma datos entre chains
    5. 🎯 CustomChain: Chains personalizadas
    
    💡 VENTAJAS:
    - Modularidad y reutilización
    - Flujos complejos y organizados
    - Fácil debugging y testing
    - Composición flexible
    """)

def llm_chain_basica():
    """
    🔄 LLMChain básica
    """
    print_step(1, "LLMChain Básica", "Creando tu primera chain")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7,
            max_tokens=150
        )
        
        # Crear prompt template
        prompt = PromptTemplate(
            input_variables=["concepto"],
            template="Explica qué es {concepto} en el contexto de programación. Máximo 2 frases."
        )
        
        # Crear LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print_result("LLMChain Creada", chain)
        
        # Usar la chain
        conceptos = ["API", "JSON", "REST"]
        
        print("\n🔍 Probando la chain:")
        for concepto in conceptos:
            resultado = chain.run(concepto)
            print(f"📝 {concepto}: {resultado}")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def llm_chain_con_variables():
    """
    🔧 LLMChain con múltiples variables
    """
    print_step(2, "LLMChain con Variables", "Chains más complejas")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Prompt con múltiples variables
        prompt = PromptTemplate(
            input_variables=["lenguaje", "concepto", "nivel"],
            template="""
            Eres un instructor de programación experto en {lenguaje}.
            Explica el concepto de {concepto} a nivel {nivel}.
            Mantén la explicación clara y concisa.
            """
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print_result("Chain con Variables", chain)
        
        # Ejemplos de uso
        ejemplos = [
            {"lenguaje": "Python", "concepto": "decoradores", "nivel": "intermedio"},
            {"lenguaje": "JavaScript", "concepto": "closures", "nivel": "avanzado"},
            {"lenguaje": "Java", "concepto": "polimorfismo", "nivel": "básico"}
        ]
        
        print("\n🔍 Probando con diferentes variables:")
        for ejemplo in ejemplos:
            resultado = chain.run(ejemplo)
            print(f"\n📝 {ejemplo['lenguaje']} - {ejemplo['concepto']} ({ejemplo['nivel']}):")
            print(f"   {resultado}")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def simple_sequential_chain():
    """
    🔗 SimpleSequentialChain - Chains en secuencia
    """
    print_step(3, "SimpleSequentialChain", "Ejecutando chains en secuencia")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Chain 1: Generar concepto
        prompt1 = PromptTemplate(
            input_variables=["tema"],
            template="Genera un concepto de programación relacionado con {tema}. Solo el nombre del concepto."
        )
        chain1 = LLMChain(llm=llm, prompt=prompt1)
        
        # Chain 2: Explicar concepto
        prompt2 = PromptTemplate(
            input_variables=["concepto"],
            template="Explica qué es {concepto} en programación. Máximo 2 frases."
        )
        chain2 = LLMChain(llm=llm, prompt=prompt2)
        
        # Crear sequential chain
        sequential_chain = SimpleSequentialChain(
            chains=[chain1, chain2],
            verbose=True
        )
        
        print_result("Sequential Chain", sequential_chain)
        
        # Probar la chain
        temas = ["estructuras de datos", "algoritmos", "patrones de diseño"]
        
        print("\n🔗 Probando sequential chain:")
        for tema in temas:
            resultado = sequential_chain.run(tema)
            print(f"\n📝 Tema: {tema}")
            print(f"   Resultado: {resultado}")
        
        return sequential_chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def sequential_chain_avanzada():
    """
    🔗 SequentialChain avanzada con múltiples inputs/outputs
    """
    print_step(4, "SequentialChain Avanzada", "Chains con múltiples variables")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Chain 1: Analizar concepto
        prompt1 = PromptTemplate(
            input_variables=["concepto"],
            template="""
            Analiza el concepto de programación: {concepto}
            
            Proporciona:
            - Definición: Una definición clara
            - Ejemplo: Un ejemplo práctico
            - Dificultad: Básico, Intermedio o Avanzado
            """
        )
        chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="analisis")
        
        # Chain 2: Crear ejercicio
        prompt2 = PromptTemplate(
            input_variables=["concepto", "analisis"],
            template="""
            Basándote en el análisis de {concepto}:
            {analisis}
            
            Crea un ejercicio práctico para practicar este concepto.
            """
        )
        chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="ejercicio")
        
        # Chain 3: Generar solución
        prompt3 = PromptTemplate(
            input_variables=["concepto", "ejercicio"],
            template="""
            Para el concepto {concepto} y el ejercicio:
            {ejercicio}
            
            Proporciona una solución paso a paso.
            """
        )
        chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="solucion")

        
        # Crear sequential chain
        sequential_chain = SequentialChain(
            chains=[chain1, chain2, chain3],
            input_variables=["concepto"],
            output_variables=["analisis", "ejercicio", "solucion"],
            verbose=True
        )
        
        # Contexto inicial: {"concepto": "funciones lambda"} (proporcionado por input_variables).
        # Ejecutar chain1: su prompt1 usa input_variables=["concepto"], devuelve texto y lo guarda en el contexto bajo analisis (porque definiste output_key="analisis" en LLMChain).→ Contexto ahora: {"concepto": "...", "analisis": "texto del análisis..."}
        # Ejecutar chain2: su prompt2 declara input_variables=["concepto", "analisis"], por lo tanto toma ambos del contexto, genera ejercicio y lo guarda.→ Contexto ahora incluye "ejercicio": "texto del ejercicio..."
        # Ejecutar chain3: su prompt3 necesita ["concepto", "ejercicio"], genera solucion y la guarda.→ Contexto final: {"concepto": "...", "analisis": "...", "ejercicio": "...", "solucion": "..."}
        # Resultado devuelto: solo las claves listadas en output_variables (aquí analisis, ejercicio, solucion).
        
        print_result("Sequential Chain Avanzada", sequential_chain)
        
        # Probar la chain
        conceptos = ["funciones lambda", "list comprehensions"]
        
        print("\n🔗 Probando sequential chain avanzada:")
        for concepto in conceptos:
            resultado = sequential_chain({"concepto": concepto})
            print(f"\n📝 Concepto: {concepto}")
            print(f"   Análisis: {resultado['analisis'][:100]}...")
            print(f"   Ejercicio: {resultado['ejercicio'][:100]}...")
            print(f"   Solución: {resultado['solucion'][:100]}...")
        
        return sequential_chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def chain_con_output_parser():
    """
    🔧 Chain con Output Parser
    """
    print_step(5, "Chain con Output Parser", "Generando respuestas estructuradas")
    
    try:
        from langchain.output_parsers import ResponseSchema, StructuredOutputParser
        
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Definir esquema de respuesta
        response_schemas = [
            ResponseSchema(name="concepto", description="El concepto analizado", type="string"),
            ResponseSchema(name="definicion", description="Definición clara", type="string"),
            ResponseSchema(name="ejemplo", description="Ejemplo de código", type="string"),
            ResponseSchema(name="dificultad", description="Nivel de dificultad", type="string"),
            ResponseSchema(name="aplicaciones", description="Casos de uso comunes", type="string")
        ]
        
        # Crear parser
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        # Prompt con formato de salida
        prompt = PromptTemplate(
            template="Analiza el concepto de programación: {concepto}\n{format_instructions}",
            input_variables=["concepto"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print_result("Chain con Parser", chain)
        
        # Probar la chain
        conceptos = ["decoradores", "generadores"]
        
        print("\n🔧 Probando chain con parser:")
        for concepto in conceptos:
            resultado = chain.run(concepto)
            print(f"\n📝 Concepto: {concepto}")
            print(f"   Resultado: {resultado}")
            
            # Parsear resultado
            try:
                parsed = output_parser.parse(resultado)
                print(f"   Parseado: {parsed}")
            except Exception as e:
                print(f"   Error al parsear: {e}")
        
        return chain, output_parser
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def chain_con_callbacks():
    """
    📊 Chain con Callbacks para monitoreo
    """
    print_step(6, "Chain con Callbacks", "Monitoreando el rendimiento")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Prompt template
        prompt = PromptTemplate(
            input_variables=["tema"],
            template="Explica el tema de programación: {tema}. Máximo 3 frases."
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        print_result("Chain con Callbacks", chain)
        
        # Usar callbacks para monitoreo
        temas = ["variables", "funciones", "clases"]
        
        print("\n📊 Monitoreando con callbacks:")
        for tema in temas:
            with get_openai_callback() as cb:
                resultado = chain.run(tema)
                
                print(f"\n📝 Tema: {tema}")
                print(f"   Resultado: {resultado}")
                print(f"   Tokens totales: {cb.total_tokens}")
                print(f"   Tokens prompt: {cb.prompt_tokens}")
                print(f"   Tokens respuesta: {cb.completion_tokens}")
                print(f"   Costo: ${cb.total_cost}")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def manejo_errores_chains():
    """
    🛡️ Manejo de errores en chains
    """
    print_step(7, "Manejo de Errores", "Creando chains robustas")
    
    print("""
    🛡️ ESTRATEGIAS DE MANEJO DE ERRORES:
    
    1. 🔄 Retry Logic: Reintentar en caso de fallo
    2. 🛡️ Try-Catch: Capturar excepciones específicas
    3. 🔀 Fallback: Alternativas cuando falla la chain principal
    4. ⚠️ Validation: Validar inputs antes de procesar
    5. 📊 Logging: Registrar errores para debugging
    """)
    
    try:
        # Crear LLM con timeout
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7,
            request_timeout=10
        )
        
        # Prompt template
        prompt = PromptTemplate(
            input_variables=["concepto"],
            template="Explica qué es {concepto} en programación."
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Función con manejo de errores
        def ejecutar_chain_segura(concepto):
            try:
                resultado = chain.run(concepto)
                return {"success": True, "result": resultado}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Probar con manejo de errores
        conceptos = ["API", "invalid_concept_123", "JSON"]
        
        print("\n🛡️ Probando manejo de errores:")
        for concepto in conceptos:
            resultado = ejecutar_chain_segura(concepto)
            if resultado["success"]:
                print(f"✅ {concepto}: {resultado['result']}")
            else:
                print(f"❌ {concepto}: Error - {resultado['error']}")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def optimizacion_rendimiento():
    """
    ⚡ Optimización de rendimiento en chains
    """
    print_step(8, "Optimización de Rendimiento", "Mejorando la eficiencia")
    
    print("""
    ⚡ ESTRATEGIAS DE OPTIMIZACIÓN:
    
    1. 🔄 Caching: Guardar resultados para reutilizar
    2. ⚡ Batch Processing: Procesar múltiples inputs juntos
    3. 🎯 Model Selection: Usar modelos apropiados para cada tarea
    4. 📏 Token Optimization: Minimizar tokens innecesarios
    5. 🔀 Parallel Processing: Ejecutar chains en paralelo cuando sea posible
    """)
    
    try:
        import time
        
        # Crear LLM optimizado
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Modelo más rápido
            temperature=0.7,
            max_tokens=100  # Limitar tokens
        )
        
        # Prompt optimizado
        prompt = PromptTemplate(
            input_variables=["concepto"],
            template="Explica {concepto} brevemente."  # Prompt corto
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Medir rendimiento
        conceptos = ["variables", "funciones", "clases", "módulos"]
        
        print("\n⚡ Probando optimización:")
        start_time = time.time()
        
        for concepto in conceptos:
            chain_start = time.time()
            resultado = chain.run(concepto)
            chain_time = time.time() - chain_start
            
            print(f"📝 {concepto}: {resultado[:50]}... (Tiempo: {chain_time:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Tiempo total: {total_time:.2f} segundos")
        print(f"📊 Promedio por concepto: {total_time/len(conceptos):.2f} segundos")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def ejercicios_practicos():
    """
    🎯 Ejercicios prácticos
    """
    print_step(9, "Ejercicios Prácticos", "Pon en práctica lo aprendido")
    
    ejercicios = [
        {
            "titulo": "Chain de Análisis de Código",
            "descripcion": "Crea una chain que analice código Python y genere un reporte",
            "objetivo": "Practicar LLMChain con output parser"
        },
        {
            "titulo": "Chain de Traducción",
            "descripcion": "Crea una sequential chain que traduzca texto y luego lo resuma",
            "objetivo": "Aprender SequentialChain"
        },
        {
            "titulo": "Chain con Validación",
            "descripcion": "Crea una chain que valide inputs antes de procesarlos",
            "objetivo": "Practicar manejo de errores"
        },
        {
            "titulo": "Chain de Generación de Contenido",
            "descripcion": "Crea una chain que genere contenido educativo paso a paso",
            "objetivo": "Aprender chain composition"
        },
        {
            "titulo": "Chain Optimizada",
            "descripcion": "Optimiza una chain existente para mejor rendimiento",
            "objetivo": "Practicar optimización"
        }
    ]
    
    print("\n🎯 EJERCICIOS PRÁCTICOS:")
    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{i}. {ejercicio['titulo']}")
        print(f"   Descripción: {ejercicio['descripcion']}")
        print(f"   Objetivo: {ejercicio['objetivo']}")

def mejores_practicas():
    """
    ⭐ Mejores prácticas para chains
    """
    print_step(10, "Mejores Prácticas", "Consejos para crear chains efectivas")
    
    practicas = [
        "🔗 Mantén las chains simples y modulares",
        "📏 Usa prompts concisos para optimizar tokens",
        "🛡️ Implementa manejo de errores robusto",
        "📊 Monitorea el rendimiento con callbacks",
        "🔄 Usa caching para respuestas frecuentes",
        "⚡ Optimiza el modelo según la tarea",
        "🔀 Considera parallel processing cuando sea posible",
        "📝 Documenta tus chains y sus propósitos",
        "🧪 Prueba tus chains con diferentes inputs",
        "📈 Mide y optimiza continuamente"
    ]
    
    print("\n⭐ MEJORES PRÁCTICAS:")
    for practica in practicas:
        print(f"   {practica}")

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 4: CHAINS SIMPLES")
    
    # Verificar configuración
    if not config.validate_config():
        return
    
    # Contenido del módulo
    introduccion_chains()
    
    # Chains básicas
    llm_chain_basica()
    llm_chain_con_variables()
    
    # Chains secuenciales
    simple_sequential_chain()
    sequential_chain_avanzada()
    
    # Características avanzadas
    chain_con_output_parser()
    chain_con_callbacks()
    
    # Robustez y optimización
    manejo_errores_chains()
    optimizacion_rendimiento()
    
    # Consolidación
    mejores_practicas()
    ejercicios_practicos()
    
    print("\n🎉 ¡Módulo 4 completado! Ahora dominas las chains básicas.")
    print("🚀 Próximo módulo: Chains Secuenciales Avanzadas")

if __name__ == "__main__":
    main()
