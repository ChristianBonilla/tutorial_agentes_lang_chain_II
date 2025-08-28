#!/usr/bin/env python3
"""
============================================================
    MÓDULO 5: CHAINS SECUENCIALES AVANZADAS
============================================================

🎯 OBJETIVOS:
- Dominar chains secuenciales complejas
- Implementar workflows condicionales
- Crear chains personalizadas
- Manejar datos estructurados
- Optimizar rendimiento de chains

📚 CONTENIDO:
1. Chains secuenciales complejas
2. Workflows condicionales
3. Chains personalizadas
4. Manejo de datos estructurados
5. Optimización de rendimiento
6. Chains con memoria
7. Chains paralelas
8. Ejemplos prácticos avanzados
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.base import Chain
from langchain.schema import BaseOutputParser, HumanMessage
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel, Field

# Importaciones locales
import sys
sys.path.append('..')
from utils.config import LangChainConfig
from utils.helpers import print_separator, print_step, print_result, measure_execution_time

# Configuración
config = LangChainConfig()

class TaskType(Enum):
    """Tipos de tareas"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"

@dataclass
class ChainResult:
    """Resultado de una chain"""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0

class DocumentProcessor(BaseModel):
    """Procesador de documentos"""
    title: str = Field(description="Título del documento")
    summary: str = Field(description="Resumen del contenido")
    key_points: List[str] = Field(description="Puntos clave")
    sentiment: str = Field(description="Sentimiento del documento")
    category: str = Field(description="Categoría del documento")

class AdvancedSequentialChain:
    """Chain secuencial avanzada con lógica condicional"""
    
    def __init__(self, chains: List[Chain], conditional_logic: Dict = None):
        self.chains = chains
        self.conditional_logic = conditional_logic or {}
        self.memory = {}
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la chain secuencial"""
        results = {}
        current_input = inputs
        
        for i, chain in enumerate(self.chains):
            try:
                # Verificar lógica condicional
                if self._should_execute_chain(i, current_input):
                    output = chain.run(current_input)
                    results[f"output_{i}"] = output
                    current_input = {**current_input, **results}
                else:
                    results[f"output_{i}"] = "Skipped due to condition"
                    
            except Exception as e:
                results[f"output_{i}"] = f"Error: {str(e)}"
        
        return results
    
    def _should_execute_chain(self, chain_index: int, inputs: Dict) -> bool:
        """Determina si una chain debe ejecutarse basado en condiciones"""
        condition_key = f"chain_{chain_index}_condition"
        if condition_key in self.conditional_logic:
            condition = self.conditional_logic[condition_key]
            return self._evaluate_condition(condition, inputs)
        return True
    
    def _evaluate_condition(self, condition: str, inputs: Dict) -> bool:
        """Evalúa una condición"""
        try:
            # Lógica simple de evaluación de condiciones
            if "length" in condition and "text" in inputs:
                text_length = len(inputs["text"])
                if ">" in condition:
                    min_length = int(condition.split(">")[1])
                    return text_length > min_length
                elif "<" in condition:
                    max_length = int(condition.split("<")[1])
                    return text_length < max_length
        except:
            pass
        return True

def chains_secuenciales_complejas():
    """
    🔗 Chains secuenciales complejas
    """
    print_step(1, "Chains Secuenciales Complejas", "Implementando workflows complejos")
    
    # Configurar LLM
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # 1. Chain de análisis de texto
    analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analiza el siguiente texto y proporciona:
        1. Tema principal
        2. Sentimiento (positivo/negativo/neutral)
        3. Puntos clave (máximo 3)
        
        Texto: {text}
        
        Responde en formato JSON:
        {{
            "tema": "tema principal",
            "sentimiento": "positivo/negativo/neutral",
            "puntos_clave": ["punto 1", "punto 2", "punto 3"]
        }}
        """
    )
    
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analisis")
    
    # 2. Chain de generación de resumen
    summary_prompt = PromptTemplate(
        input_variables=["text", "analisis"],
        template="""
        Basándote en el análisis previo, genera un resumen conciso del texto.
        
        Texto original: {text}
        Análisis: {analisis}
        
        Resumen (máximo 100 palabras):
        """
    )
    
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="resumen")
    
    # 3. Chain de recomendaciones
    recommendations_prompt = PromptTemplate(
        input_variables=["analisis", "resumen"],
        template="""
        Basándote en el análisis y resumen, proporciona 3 recomendaciones relacionadas.
        
        Análisis: {analisis}
        Resumen: {resumen}
        
        Recomendaciones:
        1. 
        2. 
        3. 
        """
    )
    
    recommendations_chain = LLMChain(llm=llm, prompt=recommendations_prompt, output_key="recomendaciones")
    
    # Crear chain secuencial
    sequential_chain = SequentialChain(
        chains=[analysis_chain, summary_chain, recommendations_chain],
        input_variables=["text"],
        output_variables=["analisis", "resumen", "recomendaciones"],
        verbose=True
    )
    
    # Probar chain
    test_text = """
    La inteligencia artificial está transformando la forma en que trabajamos y vivimos. 
    Desde asistentes virtuales hasta sistemas de recomendación, la IA está presente en 
    casi todos los aspectos de nuestra vida digital. Sin embargo, también plantea 
    importantes desafíos éticos y sociales que debemos abordar.
    """
    
    print("🧪 Probando chain secuencial compleja...")
    result = sequential_chain({"text": test_text})
    
    print_result("Resultado Chain Secuencial", result)
    
    return sequential_chain

def workflows_condicionales():
    """
    🔀 Workflows condicionales
    """
    print_step(2, "Workflows Condicionales", "Implementando lógica condicional en chains")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # Chain 1: Clasificación de contenido
    classification_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Clasifica el siguiente texto en una de estas categorías:
        - TECNOLOGIA
        - CIENCIA
        - POLITICA
        - DEPORTES
        - ENTRETENIMIENTO
        
        Texto: {text}
        
        Categoría:
        """
    )
    
    classification_chain = LLMChain(llm=llm, prompt=classification_prompt, output_key="categoria")
    
    # Chain 2: Análisis técnico (solo para tecnología)
    tech_analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analiza el siguiente texto tecnológico y identifica:
        1. Tecnologías mencionadas
        2. Tendencias identificadas
        3. Impacto potencial
        
        Texto: {text}
        
        Análisis técnico:
        """
    )
    
    tech_analysis_chain = LLMChain(llm=llm, prompt=tech_analysis_prompt, output_key="analisis_tecnico")
    
    # Chain 3: Análisis general (para otras categorías)
    general_analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Proporciona un análisis general del siguiente texto:
        1. Puntos principales
        2. Implicaciones
        3. Contexto
        
        Texto: {text}
        
        Análisis general:
        """
    )
    
    general_analysis_chain = LLMChain(llm=llm, prompt=general_analysis_prompt, output_key="analisis_general")
    
    # Configurar lógica condicional
    conditional_logic = {
        "chain_1_condition": "always",
        "chain_2_condition": "categoria == 'TECNOLOGIA'",
        "chain_3_condition": "categoria != 'TECNOLOGIA'"
    }
    
    # Crear chain condicional
    conditional_chain = AdvancedSequentialChain(
        chains=[classification_chain, tech_analysis_chain, general_analysis_chain],
        conditional_logic=conditional_logic
    )
    
    # Probar con diferentes tipos de contenido
    test_texts = [
        "OpenAI lanza GPT-5 con capacidades revolucionarias de razonamiento",
        "El equipo de fútbol local gana el campeonato por primera vez en 20 años",
        "Nuevas políticas ambientales buscan reducir emisiones de carbono"
    ]
    
    print("🧪 Probando workflows condicionales...")
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 Texto {i+1}: {text[:50]}...")
        result = conditional_chain.run({"text": text})
        print(f"🏷️ Categoría: {result.get('output_0', 'N/A')}")
        print(f"📊 Análisis: {result.get('output_1', result.get('output_2', 'N/A'))}")
    
    return conditional_chain

def chains_personalizadas():
    """
    🛠️ Chains personalizadas
    """
    print_step(3, "Chains Personalizadas", "Creando chains especializadas")
    
    class DocumentAnalysisChain:
        """Chain personalizada para análisis de documentos"""
        
        def __init__(self, llm):
            self.llm = llm
            self.parser = PydanticOutputParser(pydantic_object=DocumentProcessor)
        
        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            document_text = inputs["document_text"]
            
            # Prompt para análisis estructurado
            prompt = f"""
            Analiza el siguiente documento y proporciona información estructurada:
            
            {self.parser.get_format_instructions()}
            
            Documento: {document_text}
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                structured_data = self.parser.parse(response.content)
                
                return {
                    "analysis_result": response.content,
                    "structured_data": structured_data
                }
            except Exception as e:
                return {
                    "analysis_result": f"Error en análisis: {str(e)}",
                    "structured_data": None
                }


    class MultiLanguageChain:
        """Chain para procesamiento multiidioma"""
        
        def __init__(self, llm, target_language="es"):
            self.llm = llm
            self.target_language = target_language
        
        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            text = inputs["text"]
            source_language = inputs.get("source_language", "auto")
            
            # Detectar idioma si no se especifica
            if source_language == "auto":
                detect_prompt = f"Detecta el idioma del siguiente texto y responde solo con el código del idioma (es, en, fr, de, etc.): {text}"
                detection_response = self.llm.invoke([HumanMessage(content=detect_prompt)])
                detected_lang = detection_response.content.strip()
            else:
                detected_lang = source_language
            
            # Traducir si es necesario
            if detected_lang != self.target_language:
                translate_prompt = f"Traduce el siguiente texto del {detected_lang} al {self.target_language}: {text}"
                translation_response = self.llm.invoke([HumanMessage(content=translate_prompt)])
                translated_text = translation_response.content
            else:
                translated_text = text
            
            return {
                "translated_text": translated_text,
                "language_detected": detected_lang
            }
    
    # Probar chains personalizadas
    llm = ChatOpenAI(model=config.default_model, temperature=0.3)
    
    # Document Analysis Chain
    doc_chain = DocumentAnalysisChain(llm)
    
    test_document = """
    La inteligencia artificial está revolucionando la medicina moderna. 
    Los algoritmos de machine learning pueden detectar enfermedades con mayor 
    precisión que los médicos humanos en algunos casos. Sin embargo, esto 
    plantea importantes cuestiones éticas sobre la privacidad de los datos 
    y la responsabilidad en el diagnóstico médico.
    """
    
    print("🧪 Probando Document Analysis Chain...")
    doc_result = doc_chain.run({"document_text": test_document})
    print_result("Análisis de Documento", doc_result)
    
    # Multi Language Chain
    multi_lang_chain = MultiLanguageChain(llm, target_language="es")
    
    test_texts = [
        ("Hello, how are you today?", "en"),
        ("Bonjour, comment allez-vous?", "fr"),
        ("Hola, ¿cómo estás hoy?", "auto")
    ]
    
    print("\n🌍 Probando Multi Language Chain...")
    for text, lang in test_texts:
        result = multi_lang_chain.run({"text": text, "source_language": lang})
        print(f"📝 Original ({lang}): {text}")
        print(f"🇪🇸 Traducido: {result['translated_text']}")
        print(f"🔍 Idioma detectado: {result['language_detected']}\n")
    
    return {
        'document_analysis': doc_chain,
        'multi_language': multi_lang_chain
    }

def manejo_datos_estructurados():
    """
    📊 Manejo de datos estructurados
    """
    print_step(4, "Manejo de Datos Estructurados", "Procesando datos complejos")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.1)
    
    # Parser para listas
    list_parser = CommaSeparatedListOutputParser()
    
    # Chain para extraer entidades
    entity_extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extrae las siguientes entidades del texto:
        - Personas
        - Organizaciones
        - Lugares
        - Fechas
        
        Texto: {text}
        
        Responde en formato de lista separada por comas:
        Personas: {format_instructions}
        Organizaciones: {format_instructions}
        Lugares: {format_instructions}
        Fechas: {format_instructions}
        """.replace("{format_instructions}", list_parser.get_format_instructions())
    )
    
    entity_chain = LLMChain(llm=llm, prompt=entity_extraction_prompt)
    
    # Chain para análisis de sentimientos detallado
    sentiment_analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analiza el sentimiento del siguiente texto y proporciona:
        1. Sentimiento general (positivo/negativo/neutral)
        2. Intensidad (baja/media/alta)
        3. Emociones específicas detectadas
        4. Confianza en el análisis (0-100%)
        
        Texto: {text}
        
        Responde en formato JSON:
        {{
            "sentimiento": "positivo/negativo/neutral",
            "intensidad": "baja/media/alta",
            "emociones": ["emoción1", "emoción2"],
            "confianza": 85
        }}
        """
    )
    
    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_analysis_prompt)
    
    # Probar con texto complejo
    complex_text = """
    Apple Inc. anunció hoy en su sede de Cupertino, California, que Tim Cook, 
    CEO de la empresa, presentará nuevos productos el próximo 15 de septiembre. 
    Los analistas de Goldman Sachs y Morgan Stanley están muy entusiasmados con 
    las innovaciones esperadas, aunque algunos inversores muestran preocupación 
    por los costos de producción.
    """
    
    print("🧪 Probando extracción de entidades...")
    entity_result = entity_chain.run({"text": complex_text})
    print_result("Entidades Extraídas", entity_result)
    
    print("\n🧪 Probando análisis de sentimientos...")
    sentiment_result = sentiment_chain.run({"text": complex_text})
    print_result("Análisis de Sentimientos", sentiment_result)
    
    return {
        'entity_extraction': entity_chain,
        'sentiment_analysis': sentiment_chain
    }

def optimizacion_rendimiento():
    """
    ⚡ Optimización de rendimiento
    """
    print_step(5, "Optimización de Rendimiento", "Mejorando velocidad y eficiencia")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # Chain optimizada con caching
    class OptimizedChain:
        """Chain optimizada con caching y métricas"""
        
        def __init__(self, llm, prompt_template):
            self.llm = llm
            self.prompt_template = prompt_template
            self.cache = {}
            self.metrics = {
                'total_calls': 0,
                'cache_hits': 0,
                'total_time': 0
            }
        
        def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            self.metrics['total_calls'] += 1
            
            # Crear clave de cache
            cache_key = str(sorted(inputs.items()))
            
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                result = self.cache[cache_key]
            else:
                # Ejecutar chain
                prompt = self.prompt_template.format(**inputs)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                result = response.content
                
                # Guardar en cache
                self.cache[cache_key] = result
            
            execution_time = time.time() - start_time
            self.metrics['total_time'] += execution_time
            
            return {
                "output": result,
                "metrics": {
                    "execution_time": execution_time,
                    "cache_hit": cache_key in self.cache,
                    "total_calls": self.metrics['total_calls'],
                    "cache_hits": self.metrics['cache_hits'],
                    "avg_time": self.metrics['total_time'] / self.metrics['total_calls']
                }
            }
    
    # Crear chain optimizada
    optimization_prompt = PromptTemplate(
        input_variables=["text", "task"],
        template="""
        Realiza la siguiente tarea en el texto:
        
        Tarea: {task}
        Texto: {text}
        
        Resultado:
        """
    )
    
    optimized_chain = OptimizedChain(llm, optimization_prompt)
    
    # Probar optimización
    test_inputs = [
        {"text": "Python es un lenguaje de programación", "task": "Explicar brevemente"},
        {"text": "Python es un lenguaje de programación", "task": "Explicar brevemente"},  # Duplicado para cache
        {"text": "Machine learning es una rama de la IA", "task": "Definir concepto"},
        {"text": "Machine learning es una rama de la IA", "task": "Definir concepto"}  # Duplicado para cache
    ]
    
    print("🧪 Probando optimización de rendimiento...")
    
    for i, inputs in enumerate(test_inputs):
        print(f"\n🔄 Ejecución {i+1}: {inputs['task']}")
        result = optimized_chain.run(inputs)
        print(f"⏱️ Tiempo: {result['metrics']['execution_time']:.3f}s")
        print(f"💾 Cache hit: {result['metrics']['cache_hit']}")
        print(f"📊 Total calls: {result['metrics']['total_calls']}")
        print(f"📈 Cache hits: {result['metrics']['cache_hits']}")
        print(f"📉 Tiempo promedio: {result['metrics']['avg_time']:.3f}s")
    
    return optimized_chain

def chains_con_memoria():
    """
    🧠 Chains con memoria
    """
    print_step(6, "Chains con Memoria", "Implementando memoria en chains")
    
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # Chain con memoria de conversación
    conversation_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
        Eres un asistente conversacional inteligente. Usa el historial de la conversación
        para mantener contexto y proporcionar respuestas coherentes.
        
        Historial de la conversación:
        {history}
        
        Usuario: {input}
        Asistente:
        """
    )
    
    conversation_memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input"
    )
    
    conversation_chain = LLMChain(
        llm=llm,
        prompt=conversation_prompt,
        memory=conversation_memory,
        verbose=True
    )
    
    # Chain con memoria de resumen
    summary_prompt = PromptTemplate(
        input_variables=["summary", "input"],
        template="""
        Eres un analista que mantiene un resumen de la conversación.
        
        Resumen anterior: {summary}
        
        Nueva entrada: {input}
        
        Actualiza el resumen manteniendo la información más importante:
        """
    )
    
    summary_memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="summary",
        input_key="input"
    )
    
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        memory=summary_memory
    )
    
    # Probar chains con memoria
    conversation_inputs = [
        "Hola, me llamo Juan",
        "Me gusta la programación en Python",
        "¿Recuerdas mi nombre?",
        "¿Qué te dije sobre mis intereses?"
    ]
    
    print("🧪 Probando chain con memoria de conversación...")
    for i, user_input in enumerate(conversation_inputs):
        print(f"\n👤 Usuario: {user_input}")
        response = conversation_chain.run({"input": user_input})
        print(f"🤖 Asistente: {response}")
    
    print("\n📝 Probando chain con memoria de resumen...")
    for user_input in conversation_inputs:
        summary_response = summary_chain.run({"input": user_input})
        print(f"📊 Resumen actualizado: {summary_response}")
    
    return {
        'conversation': conversation_chain,
        'summary': summary_chain
    }

def chains_paralelas():
    """
    🔄 Chains paralelas
    """
    print_step(7, "Chains Paralelas", "Ejecutando chains en paralelo")
    
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # Definir chains especializadas
    chains = {}
    
    # Chain de análisis de sentimientos
    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template="Analiza el sentimiento del texto: {text}. Responde solo: POSITIVO, NEGATIVO, o NEUTRAL."
    )
    chains['sentiment'] = LLMChain(llm=llm, prompt=sentiment_prompt)
    
    # Chain de extracción de palabras clave
    keywords_prompt = PromptTemplate(
        input_variables=["text"],
        template="Extrae las 5 palabras clave más importantes del texto: {text}"
    )
    chains['keywords'] = LLMChain(llm=llm, prompt=keywords_prompt)
    
    # Chain de resumen
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Genera un resumen de máximo 50 palabras del texto: {text}"
    )
    chains['summary'] = LLMChain(llm=llm, prompt=summary_prompt)
    
    # Chain de clasificación
    classification_prompt = PromptTemplate(
        input_variables=["text"],
        template="Clasifica el texto en una categoría: {text}. Opciones: TECNOLOGIA, CIENCIA, POLITICA, DEPORTES, ENTRETENIMIENTO"
    )
    chains['classification'] = LLMChain(llm=llm, prompt=classification_prompt)
    
    async def execute_chain_parallel(chain_name: str, chain: LLMChain, text: str) -> Dict[str, Any]:
        """Ejecuta una chain de forma asíncrona"""
        try:
            start_time = time.time()
            result = await chain.arun(text)
            execution_time = time.time() - start_time
            
            return {
                'chain_name': chain_name,
                'result': result,
                'execution_time': execution_time,
                'status': 'success'
            }
        except Exception as e:
            return {
                'chain_name': chain_name,
                'error': str(e),
                'status': 'error'
            }
    
    async def run_parallel_analysis(text: str) -> Dict[str, Any]:
        """Ejecuta todas las chains en paralelo"""
        tasks = []
        
        for chain_name, chain in chains.items():
            task = execute_chain_parallel(chain_name, chain, text)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Organizar resultados
        organized_results = {}
        for result in results:
            organized_results[result['chain_name']] = result
        
        return organized_results
    
    # Probar chains paralelas
    test_text = """
    La inteligencia artificial está transformando la medicina moderna. 
    Los algoritmos de machine learning pueden detectar enfermedades con mayor 
    precisión que los médicos humanos en algunos casos. Esta tecnología 
    promete revolucionar el diagnóstico y tratamiento de pacientes.
    """
    
    print("🧪 Ejecutando chains en paralelo...")
    
    async def test_parallel():
        results = await run_parallel_analysis(test_text)
        
        print("\n📊 Resultados de análisis paralelo:")
        for chain_name, result in results.items():
            print(f"\n🔗 {chain_name.upper()}:")
            if result['status'] == 'success':
                print(f"   ✅ Resultado: {result['result']}")
                print(f"   ⏱️ Tiempo: {result['execution_time']:.3f}s")
            else:
                print(f"   ❌ Error: {result['error']}")
    
    # Ejecutar prueba asíncrona
    asyncio.run(test_parallel())
    
    return chains

def ejemplos_practicos_avanzados():
    """
    🚀 Ejemplos prácticos avanzados
    """
    print_step(8, "Ejemplos Prácticos Avanzados", "Implementando casos de uso complejos")
    
    # 1. Sistema de análisis de documentos empresariales
    def sistema_analisis_documentos():
        """Sistema completo de análisis de documentos"""
        
        llm = ChatOpenAI(model=config.default_model, temperature=0.3)
        
        # Chain 1: Extracción de información clave
        extraction_prompt = PromptTemplate(
            input_variables=["document"],
            template="""
            Extrae la siguiente información del documento empresarial:
            - Empresa mencionada
            - Fecha del documento
            - Valor monetario (si existe)
            - Tipo de documento
            - Personas mencionadas
            
            Documento: {document}
            
            Responde en formato JSON.
            """
        )
        
        extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)
        
        # Chain 2: Análisis de riesgo
        risk_prompt = PromptTemplate(
            input_variables=["document", "extracted_info"],
            template="""
            Basándote en la información extraída, evalúa el nivel de riesgo del documento:
            
            Información extraída: {extracted_info}
            Documento: {document}
            
            Evalúa:
            - Nivel de riesgo (BAJO/MEDIO/ALTO)
            - Razones del riesgo
            - Recomendaciones
            
            Responde en formato JSON.
            """
        )
        
        risk_chain = LLMChain(llm=llm, prompt=risk_prompt)
        
        # Probar con documento de ejemplo
        sample_document = """
        CONTRATO DE SERVICIOS
        
        Fecha: 15 de marzo de 2024
        Empresa: TechCorp Solutions
        Cliente: ABC Corporation
        
        Valor del contrato: $500,000 USD
        Duración: 12 meses
        
        Responsable del proyecto: María González
        Supervisor: Carlos Rodríguez
        
        Este contrato establece los términos para el desarrollo de un sistema
        de inteligencia artificial para análisis de datos financieros.
        """
        
        print("📄 Analizando documento empresarial...")
        
        # Ejecutar chains por separado
        extraction_result = extraction_chain.run({"document": sample_document})
        print("📋 Información extraída:", extraction_result)
        
        risk_result = risk_chain.run({"document": sample_document, "extracted_info": extraction_result})
        print("⚠️ Análisis de riesgo:", risk_result)
        
        return {
            'extraction_chain': extraction_chain,
            'risk_chain': risk_chain
        }
    
    # 2. Sistema de generación de contenido multiidioma
    def sistema_contenido_multiidioma():
        """Sistema para generar contenido en múltiples idiomas"""
        
        llm = ChatOpenAI(model=config.default_model, temperature=0.7)
        
        # Chain 1: Generación de contenido base
        content_prompt = PromptTemplate(
            input_variables=["topic", "language"],
            template="""
            Genera contenido sobre el tema '{topic}' en {language}.
            El contenido debe ser informativo y atractivo.
            
            Tema: {topic}
            Idioma: {language}
            
            Contenido:
            """
        )
        
        content_chain = LLMChain(llm=llm, prompt=content_prompt)
        
        # Chain 2: Optimización SEO
        seo_prompt = PromptTemplate(
            input_variables=["content", "language"],
            template="""
            Optimiza el siguiente contenido para SEO en {language}:
            
            Contenido original: {content}
            
            Proporciona:
            1. Título optimizado
            2. Meta descripción
            3. Palabras clave principales
            4. Contenido optimizado
            
            Responde en formato JSON.
            """
        )
        
        seo_chain = LLMChain(llm=llm, prompt=seo_prompt)
        
        # Probar con diferentes idiomas
        topics = [
            ("Inteligencia Artificial", "español"),
            ("Artificial Intelligence", "inglés"),
            ("Intelligence Artificielle", "francés")
        ]
        
        print("🌍 Generando contenido multiidioma...")
        
        for topic, language in topics:
            print(f"\n📝 Generando contenido sobre '{topic}' en {language}...")
            
            # Generar contenido
            content_result = content_chain.run({"topic": topic, "language": language})
            
            # Optimizar SEO
            seo_result = seo_chain.run({"content": content_result, "language": language})
            
            print(f"✅ Contenido generado y optimizado para {language}")
        
        return {
            'content_generation': content_chain,
            'seo_optimization': seo_chain
        }
    
    # Ejecutar ejemplos
    sistema_analisis_documentos()
    print_separator("ANÁLISIS DE DOCUMENTOS")
    sistema_contenido_multiidioma()

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 5: CHAINS SECUENCIALES AVANZADAS")
    
    # Validar configuración
    if not config.validate_config():
        print("❌ Error en configuración")
        return
    
    print("✅ Configuración validada correctamente")
    
    # Ejecutar todos los ejemplos
    chains_secuenciales_complejas()
    workflows_condicionales()
    chains_personalizadas()
    manejo_datos_estructurados()
    optimizacion_rendimiento()
    chains_con_memoria()
    chains_paralelas()
    ejemplos_practicos_avanzados()
    
    print_separator("COMPLETADO")
    print("🎉 ¡Módulo 5 completado! Ahora dominas chains secuenciales avanzadas.")
    print("🚀 Próximo módulo: Agentes Básicos")

if __name__ == "__main__":
    main()
