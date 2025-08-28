#!/usr/bin/env python3
"""
============================================================
    MÓDULO 4: LLMs AVANZADOS Y PROVEEDORES MÚLTIPLES
============================================================

🎯 OBJETIVOS:
- Dominar diferentes proveedores de LLMs
- Implementar streaming y operaciones asíncronas
- Configurar parámetros avanzados
- Manejar errores y fallbacks
- Optimizar costos y rendimiento

📚 CONTENIDO:
1. Proveedores múltiples (OpenAI, Anthropic, Google, etc.)
2. Streaming de respuestas
3. Operaciones asíncronas
4. Configuraciones avanzadas
5. Manejo de errores y fallbacks
6. Optimización de costos
7. Comparación de modelos
8. Ejemplos prácticos avanzados
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Importaciones locales
import sys
sys.path.append('..')
from utils.config import LangChainConfig
from utils.helpers import print_separator, print_step, print_result, measure_execution_time

# Configuración
config = LangChainConfig()

class ModelResponse(BaseModel):
    """Modelo para respuestas estructuradas"""
    answer: str = Field(description="La respuesta principal")
    confidence: float = Field(description="Nivel de confianza (0-1)")
    reasoning: str = Field(description="Razonamiento detrás de la respuesta")
    sources: List[str] = Field(description="Fuentes utilizadas", default=[])

class StreamingCallback(BaseCallbackHandler):
    """Callback personalizado para streaming"""
    
    def __init__(self):
        self.tokens = []
        self.start_time = time.time()
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Se ejecuta cuando llega un nuevo token"""
        self.tokens.append(token)
        print(token, end='', flush=True)
    
    def on_llm_end(self, response, **kwargs):
        """Se ejecuta cuando termina la generación"""
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"\n\n⏱️ Tiempo total: {duration:.2f}s")
        print(f"📊 Tokens generados: {len(self.tokens)}")

def configurar_proveedores():
    """
    🔧 Configuración de múltiples proveedores de LLMs
    """
    print_step(1, "Configuración de Proveedores", "Configurando diferentes proveedores de LLMs")
    
    providers = {}
    
    # OpenAI
    try:
        providers['openai'] = {
            'gpt-4': ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                max_tokens=1000,
                streaming=True
            ),
            'gpt-3.5-turbo': ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=500
            )
        }
        print("✅ OpenAI configurado")
    except Exception as e:
        print(f"❌ Error configurando OpenAI: {e}")
    
    # Anthropic (Claude)
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            providers['anthropic'] = {
                'claude-3-sonnet': ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.7,
                    max_tokens=1000
                ),
                'claude-3-haiku': ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.5,
                    max_tokens=500
                )
            }
            print("✅ Anthropic configurado")
        else:
            print("⚠️ Anthropic no configurado (falta API key)")
    except Exception as e:
        print(f"❌ Error configurando Anthropic: {e}")
    
    # Google (Gemini)
    try:
        if os.getenv("GOOGLE_API_KEY"):
            providers['google'] = {
                'gemini-pro': ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0.7,
                    max_output_tokens=1000
                )
            }
            print("✅ Google configurado")
        else:
            print("⚠️ Google no configurado (falta API key)")
    except Exception as e:
        print(f"❌ Error configurando Google: {e}")
    
    return providers

def streaming_example():
    """
    🌊 Ejemplo de streaming de respuestas
    """
    print_step(2, "Streaming de Respuestas", "Implementando streaming en tiempo real")
    
    # Configurar callback para streaming
    streaming_handler = StreamingCallback()
    
    # LLM con streaming
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        streaming=True,
        callbacks=[streaming_handler]
    )
    
    prompt = """
    Escribe una historia corta de ciencia ficción sobre un viajero del tiempo
    que descubre que puede cambiar el pasado, pero cada cambio tiene consecuencias
    inesperadas. La historia debe ser emocionante y tener un giro inesperado al final.
    """
    
    print("🚀 Iniciando streaming...")
    print("📝 Historia generada:")
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"\n\n✅ Streaming completado")
        return response
    except Exception as e:
        print(f"❌ Error en streaming: {e}")
        return None

async def operaciones_asincronas():
    """
    ⚡ Operaciones asíncronas con múltiples LLMs
    """
    print_step(3, "Operaciones Asíncronas", "Ejecutando múltiples LLMs en paralelo")
    
    providers = configurar_proveedores()
    
    # Pregunta para comparar
    question = "¿Cuáles son las principales diferencias entre la inteligencia artificial y la inteligencia humana?"
    
    tasks = []
    
    # Crear tareas asíncronas para cada proveedor
    for provider_name, models in providers.items():
        for model_name, llm in models.items():
            task = asyncio.create_task(
                llm.ainvoke([HumanMessage(content=question)])
            )
            tasks.append((provider_name, model_name, task))
    
    print(f"🔄 Ejecutando {len(tasks)} modelos en paralelo...")
    
    results = {}
    
    # Esperar todas las respuestas
    for provider_name, model_name, task in tasks:
        try:
            response = await task
            results[f"{provider_name}_{model_name}"] = {
                'provider': provider_name,
                'model': model_name,
                'response': response.content,
                'status': 'success'
            }
            print(f"✅ {provider_name} - {model_name}: Completado")
        except Exception as e:
            results[f"{provider_name}_{model_name}"] = {
                'provider': provider_name,
                'model': model_name,
                'error': str(e),
                'status': 'error'
            }
            print(f"❌ {provider_name} - {model_name}: Error - {e}")
    
    return results

def configuraciones_avanzadas():
    """
    ⚙️ Configuraciones avanzadas de LLMs
    """
    print_step(4, "Configuraciones Avanzadas", "Implementando configuraciones especializadas")
    
    # 1. LLM con función de llamada
    llm_function = ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        functions=[
            {
                "name": "get_weather",
                "description": "Obtener información del clima",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "La ciudad y estado, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    )
    
    # 2. LLM con parser estructurado
    parser = PydanticOutputParser(pydantic_object=ModelResponse)
    
    llm_structured = ChatOpenAI(
        model="gpt-4",
        temperature=0.1
    )
    
    # 3. LLM con configuración de seguridad
    llm_safe = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        presence_penalty=0.1,
        frequency_penalty=0.1
    )
    
    print("✅ Configuraciones avanzadas creadas")
    
    # Probar parser estructurado
    prompt = f"""
    Analiza la siguiente pregunta y proporciona una respuesta estructurada:
    
    Pregunta: ¿Cuáles son los beneficios de la inteligencia artificial?
    
    {parser.get_format_instructions()}
    """
    
    try:
        response = llm_structured.invoke([HumanMessage(content=prompt)])
        parsed_response = parser.parse(response.content)
        print_result("Respuesta Estructurada", parsed_response)
    except Exception as e:
        print(f"❌ Error en parser estructurado: {e}")
    
    return {
        'function_calling': llm_function,
        'structured': llm_structured,
        'safe': llm_safe
    }

def manejo_errores_fallbacks():
    """
    🛡️ Manejo de errores y fallbacks
    """
    print_step(5, "Manejo de Errores", "Implementando sistema de fallbacks")
    
    class FallbackLLM:
        """Sistema de fallback con múltiples LLMs"""
        
        def __init__(self, primary_llm, fallback_llms):
            self.primary_llm = primary_llm
            self.fallback_llms = fallback_llms
            self.current_llm = primary_llm
        
        def invoke_with_fallback(self, messages, max_retries=3):
            """Invoca LLM con sistema de fallback"""
            
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Intento {attempt + 1} con {self.current_llm.__class__.__name__}")
                    response = self.current_llm.invoke(messages)
                    print(f"✅ Respuesta exitosa")
                    return response
                except Exception as e:
                    print(f"❌ Error en intento {attempt + 1}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Cambiar a siguiente LLM
                        if self.fallback_llms:
                            self.current_llm = self.fallback_llms.pop(0)
                            print(f"🔄 Cambiando a fallback: {self.current_llm.__class__.__name__}")
                        else:
                            print("❌ No hay más fallbacks disponibles")
                            break
            
            raise Exception("Todos los LLMs fallaron")
    
    # Configurar sistema de fallback
    primary = ChatOpenAI(model="gpt-4", temperature=0.7)
    fallback1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    fallback2 = OpenAI(model="text-davinci-003", temperature=0.7)
    
    fallback_system = FallbackLLM(primary, [fallback1, fallback2])
    
    # Probar sistema
    test_message = "Explica el concepto de machine learning de manera simple"
    
    try:
        response = fallback_system.invoke_with_fallback([HumanMessage(content=test_message)])
        print_result("Respuesta con Fallback", response.content)
    except Exception as e:
        print(f"❌ Error final: {e}")
    
    return fallback_system

def optimizacion_costos():
    """
    💰 Optimización de costos
    """
    print_step(6, "Optimización de Costos", "Implementando estrategias de optimización")
    
    class CostOptimizer:
        """Optimizador de costos para LLMs"""
        
        def __init__(self):
            self.cost_tracker = {}
            self.usage_stats = {}
        
        def estimate_cost(self, model, tokens):
            """Estima el costo basado en el modelo y tokens"""
            # Precios aproximados por 1K tokens (actualizar según precios reales)
            prices = {
                'gpt-4': 0.03,  # $0.03 por 1K tokens
                'gpt-3.5-turbo': 0.002,  # $0.002 por 1K tokens
                'text-davinci-003': 0.02,  # $0.02 por 1K tokens
            }
            
            model_name = model.model_name if hasattr(model, 'model_name') else str(model)
            price_per_1k = prices.get(model_name, 0.01)
            
            return (tokens / 1000) * price_per_1k
        
        def optimize_for_cost(self, task_complexity, budget):
            """Selecciona el mejor modelo según complejidad y presupuesto"""
            
            if task_complexity == 'simple' and budget == 'low':
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
            elif task_complexity == 'medium':
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            elif task_complexity == 'complex' and budget == 'high':
                return ChatOpenAI(model="gpt-4", temperature=0.7)
            else:
                return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    optimizer = CostOptimizer()
    
    # Ejemplos de optimización
    tasks = [
        ("simple", "low", "Responde sí o no: ¿Es Python un lenguaje de programación?"),
        ("medium", "medium", "Explica brevemente qué es la inteligencia artificial"),
        ("complex", "high", "Escribe un ensayo detallado sobre el futuro de la IA")
    ]
    
    print("💰 Optimización de costos:")
    for complexity, budget, task in tasks:
        llm = optimizer.optimize_for_cost(complexity, budget)
        estimated_cost = optimizer.estimate_cost(llm, 100)  # Estimación para 100 tokens
        
        print(f"\n📊 Tarea: {complexity} | Presupuesto: {budget}")
        print(f"🤖 Modelo seleccionado: {llm.model_name}")
        print(f"💰 Costo estimado: ${estimated_cost:.4f}")
    
    return optimizer

def comparacion_modelos():
    """
    📊 Comparación de modelos
    """
    print_step(7, "Comparación de Modelos", "Evaluando diferentes modelos")
    
    providers = configurar_proveedores()
    
    # Métricas de evaluación
    test_prompts = [
        "Explica qué es la inteligencia artificial en una frase",
        "Escribe un poema corto sobre la tecnología",
        "Resuelve: Si tengo 5 manzanas y como 2, ¿cuántas me quedan?",
        "Describe el proceso de fotosíntesis"
    ]
    
    results = {}
    
    for provider_name, models in providers.items():
        for model_name, llm in models.items():
            print(f"\n🧪 Probando {provider_name} - {model_name}")
            
            model_results = []
            start_time = time.time()
            
            for i, prompt in enumerate(test_prompts):
                try:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    
                    # Métricas básicas
                    response_time = time.time() - start_time
                    token_count = len(response.content.split())
                    
                    model_results.append({
                        'prompt': prompt,
                        'response': response.content,
                        'response_time': response_time,
                        'token_count': token_count,
                        'status': 'success'
                    })
                    
                    print(f"  ✅ Prompt {i+1}: {response_time:.2f}s, {token_count} tokens")
                    
                except Exception as e:
                    model_results.append({
                        'prompt': prompt,
                        'error': str(e),
                        'status': 'error'
                    })
                    print(f"  ❌ Prompt {i+1}: Error - {e}")
            
            results[f"{provider_name}_{model_name}"] = model_results
    
    # Análisis de resultados
    print("\n📊 RESUMEN DE COMPARACIÓN:")
    print("=" * 50)
    
    for model_key, model_results in results.items():
        successful_responses = [r for r in model_results if r['status'] == 'success']
        
        if successful_responses:
            avg_time = sum(r['response_time'] for r in successful_responses) / len(successful_responses)
            avg_tokens = sum(r['token_count'] for r in successful_responses) / len(successful_responses)
            success_rate = len(successful_responses) / len(model_results) * 100
            
            print(f"\n🤖 {model_key}:")
            print(f"   ⏱️ Tiempo promedio: {avg_time:.2f}s")
            print(f"   📊 Tokens promedio: {avg_tokens:.1f}")
            print(f"   ✅ Tasa de éxito: {success_rate:.1f}%")
    
    return results

def ejemplos_practicos_avanzados():
    """
    🚀 Ejemplos prácticos avanzados
    """
    print_step(8, "Ejemplos Prácticos Avanzados", "Implementando casos de uso complejos")
    
    # 1. Sistema de análisis de sentimientos
    def analisis_sentimientos():
        """Análisis de sentimientos con múltiples modelos"""
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        
        textos = [
            "Me encanta este producto, es increíble!",
            "No estoy satisfecho con el servicio",
            "Es aceptable, pero podría ser mejor",
            "¡Terrible experiencia, no lo recomiendo!"
        ]
        
        print("😊 Análisis de Sentimientos:")
        
        for texto in textos:
            prompt = f"""
            Analiza el sentimiento del siguiente texto y responde solo con:
            - POSITIVO
            - NEGATIVO
            - NEUTRAL
            
            Texto: "{texto}"
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            print(f"📝 '{texto}' → {response.content.strip()}")
    
    # 2. Generador de código
    def generador_codigo():
        """Generador de código con validación"""
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        
        prompt = """
        Escribe una función en Python que:
        1. Tome una lista de números
        2. Encuentre el número más grande
        3. Encuentre el número más pequeño
        4. Calcule el promedio
        5. Retorne un diccionario con estos valores
        
        Solo escribe el código, sin explicaciones.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        print("💻 Código Generado:")
        print(response.content)
    
    # 3. Traductor inteligente
    def traductor_inteligente():
        """Traductor que preserva el contexto"""
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        
        textos = [
            "Hello, how are you today?",
            "The weather is beautiful today",
            "I love programming in Python"
        ]
        
        print("🌍 Traducción Inteligente:")
        
        for texto in textos:
            prompt = f"""
            Traduce el siguiente texto al español, manteniendo el tono y contexto:
            
            "{texto}"
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            print(f"🇺🇸 {texto}")
            print(f"🇪🇸 {response.content.strip()}\n")
    
    # Ejecutar ejemplos
    analisis_sentimientos()
    print_separator("ANÁLISIS DE SENTIMIENTOS")
    generador_codigo()
    print_separator("GENERADOR DE CÓDIGO")
    traductor_inteligente()

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 4: LLMs AVANZADOS Y PROVEEDORES MÚLTIPLES")
    
    # Validar configuración
    if not config.validate_config():
        print("❌ Error en configuración")
        return
    
    print("✅ Configuración validada correctamente")
    
    # Ejecutar todos los ejemplos
    providers = configurar_proveedores()
    streaming_example()
    
    # Operaciones asíncronas
    print("\n" + "="*60)
    print("🔄 Ejecutando operaciones asíncronas...")
    async_results = asyncio.run(operaciones_asincronas())
    
    configuraciones_avanzadas()
    manejo_errores_fallbacks()
    optimizacion_costos()
    comparacion_modelos()
    ejemplos_practicos_avanzados()
    
    print_separator("COMPLETADO")
    print("🎉 ¡Módulo 4 completado! Ahora dominas LLMs avanzados.")
    print("🚀 Próximo módulo: Prompts Avanzados")

if __name__ == "__main__":
    main()
