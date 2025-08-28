#!/usr/bin/env python3
"""
MÓDULO 6: AGENTES BÁSICOS
==========================

Este módulo cubre los fundamentos de los agentes en LangChain:
- Tipos de agentes
- Herramientas básicas
- Configuración de agentes
- Agentes con memoria
- Agentes especializados
- Debugging y monitoreo
- Agentes multi-herramienta
- Agentes con validación
- Agentes asíncronos
- Casos de uso prácticos

Autor: Asistente IA
Fecha: 2024
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Configuración y utilidades
from utils.config import config
from utils.helpers import print_separator, print_step, print_result, measure_execution_time

# LangChain imports
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.tools import BaseTool, tool
from langchain.tools.base import BaseTool
from langchain_community.callbacks.manager import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LLMs
from langchain_openai import ChatOpenAI

# Vector stores
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain_openai import OpenAIEmbeddings

# Document loaders
from langchain_community.document_loaders import TextLoader

# Pydantic para validación
from pydantic import BaseModel, Field, validator

# Pinecone
import pinecone

# Configurar Pinecone
from utils.pinecone_config import PineconeConfig

# ============================================================================
# CLASES Y MODELOS DE DATOS
# ============================================================================

@dataclass
class AgentResult:
    """Resultado de la ejecución de un agente"""
    success: bool
    output: str
    steps_taken: int
    execution_time: float
    tools_used: List[str]
    error_message: Optional[str] = None

class AgentMetrics(BaseModel):
    """Métricas de rendimiento del agente"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_execution_time: float = 0.0
    tools_usage: Dict[str, int] = Field(default_factory=dict)
    memory_usage: int = 0
    
    def update_metrics(self, result: AgentResult):
        """Actualizar métricas con un nuevo resultado"""
        self.total_calls += 1
        if result.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        # Actualizar tiempo promedio
        total_time = self.average_execution_time * (self.total_calls - 1) + result.execution_time
        self.average_execution_time = total_time / self.total_calls
        
        # Actualizar uso de herramientas
        for tool in result.tools_used:
            self.tools_usage[tool] = self.tools_usage.get(tool, 0) + 1

class AgentCallbackHandler(BaseCallbackHandler):
    """Callback personalizado para monitorear agentes"""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
        
    def on_agent_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Cuando el agente inicia"""
        print(f"🤖 Agente iniciado con inputs: {list(inputs.keys())}")
        
    def on_agent_action(self, action, **kwargs):
        """Cuando el agente ejecuta una acción"""
        self.current_step += 1
        step_info = {
            'step': self.current_step,
            'tool': action.tool,
            'input': action.tool_input,
            'timestamp': datetime.now().isoformat()
        }
        self.steps.append(step_info)
        print(f"🔧 Paso {self.current_step}: Usando herramienta '{action.tool}'")
        
    def on_agent_end(self, output, **kwargs):
        """Cuando el agente termina"""
        print(f"✅ Agente completado en {self.current_step} pasos")
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Cuando una herramienta inicia"""
        print(f"⚙️ Ejecutando herramienta: {serialized.get('name', 'Unknown')}")
        
    def on_tool_end(self, output: str, **kwargs):
        """Cuando una herramienta termina"""
        print(f"✅ Herramienta completada: {output[:100]}...")

# ============================================================================
# HERRAMIENTAS PERSONALIZADAS
# ============================================================================

@tool
def calcular_edad(fecha_nacimiento: str) -> str:
    """
    Calcula la edad basada en una fecha de nacimiento.
    
    Args:
        fecha_nacimiento: Fecha en formato YYYY-MM-DD
        
    Returns:
        Edad calculada en años
    """
    try:
        from datetime import datetime
        fecha = datetime.strptime(fecha_nacimiento, "%Y-%m-%d")
        hoy = datetime.now()
        edad = hoy.year - fecha.year - ((hoy.month, hoy.day) < (fecha.month, fecha.day))
        return f"La edad calculada es {edad} años"
    except Exception as e:
        return f"Error al calcular la edad: {str(e)}"

@tool
def convertir_moneda(conversion: str) -> str:
    """
    Convierte una cantidad de una moneda a otra usando tasas aproximadas.
    Formato: "cantidad moneda_origen a moneda_destino" (ej: "100 USD a EUR")
    
    Args:
        conversion: String con formato "cantidad moneda_origen a moneda_destino"
        
    Returns:
        Resultado de la conversión
    """
    try:
        # Parsear el string de entrada
        parts = conversion.split()
        if len(parts) != 4 or parts[2] != 'a':
            return "Error: Formato debe ser 'cantidad moneda_origen a moneda_destino'"
        
        cantidad = float(parts[0])
        moneda_origen = parts[1].upper()
        moneda_destino = parts[3].upper()
        
        # Tasas de cambio aproximadas
        tasas = {
            'USD': {'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0},
            'EUR': {'USD': 1.18, 'GBP': 0.86, 'JPY': 129.0},
            'GBP': {'USD': 1.37, 'EUR': 1.16, 'JPY': 150.0}
        }
        
        if moneda_origen == moneda_destino:
            return f"{cantidad} {moneda_origen} = {cantidad} {moneda_destino}"
        
        if moneda_origen in tasas and moneda_destino in tasas[moneda_origen]:
            resultado = cantidad * tasas[moneda_origen][moneda_destino]
            return f"{cantidad} {moneda_origen} = {resultado:.2f} {moneda_destino}"
        else:
            return f"No tengo tasas de cambio para {moneda_origen} a {moneda_destino}"
    except Exception as e:
        return f"Error en la conversión: {str(e)}"

@tool
def analizar_texto(texto: str) -> str:
    """
    Analiza un texto y proporciona estadísticas básicas.
    
    Args:
        texto: Texto a analizar
        
    Returns:
        Análisis del texto
    """
    try:
        palabras = texto.split()
        caracteres = len(texto)
        palabras_unicas = len(set(palabras))
        
        # Detectar idioma aproximado
        vocales_es = 'aeiouáéíóúü'
        vocales_en = 'aeiou'
        
        vocales_es_count = sum(1 for c in texto.lower() if c in vocales_es)
        vocales_en_count = sum(1 for c in texto.lower() if c in vocales_en)
        
        idioma = "Español" if vocales_es_count > vocales_en_count else "Inglés"
        
        return f"""
Análisis del texto:
- Caracteres: {caracteres}
- Palabras: {len(palabras)}
- Palabras únicas: {palabras_unicas}
- Idioma detectado: {idioma}
- Densidad de vocales: {vocales_es_count/caracteres:.2%}
        """.strip()
    except Exception as e:
        return f"Error al analizar el texto: {str(e)}"

class CalculadoraAvanzada(BaseTool):
    """Herramienta de calculadora avanzada"""
    
    name: str = "calculadora_avanzada"
    description: str = "Realiza cálculos matemáticos complejos incluyendo estadísticas"
    
    def _run(self, expresion: str) -> str:
        """Ejecuta el cálculo"""
        try:
            # Evaluar expresión matemática
            resultado = eval(expresion)
            return f"Resultado: {resultado}"
        except Exception as e:
            return f"Error en el cálculo: {str(e)}"
    
    async def _arun(self, expresion: str) -> str:
        """Versión asíncrona"""
        return self._run(expresion)

class BuscadorWeb(BaseTool):
    """Herramienta de búsqueda web simulada"""
    
    name: str = "buscador_web"
    description: str = "Busca información en la web (simulado)"
    
    def _run(self, query: str) -> str:
        """Simula una búsqueda web"""
        # Simular resultados de búsqueda
        resultados = {
            "langchain": "LangChain es un framework para desarrollar aplicaciones con LLMs",
            "python": "Python es un lenguaje de programación interpretado y de alto nivel",
            "openai": "OpenAI es una empresa de investigación en inteligencia artificial",
            "machine learning": "Machine Learning es una rama de la inteligencia artificial"
        }
        
        query_lower = query.lower()
        for key, value in resultados.items():
            if key in query_lower:
                return f"Resultado para '{query}': {value}"
        
        return f"No se encontraron resultados específicos para '{query}'. Intenta con términos como 'langchain', 'python', 'openai', o 'machine learning'."
    
    async def _arun(self, query: str) -> str:
        """Versión asíncrona"""
        return self._run(query)

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def agentes_basicos():
    """Demuestra los tipos básicos de agentes"""
    print_step("Agentes Básicos", "Explorando diferentes tipos de agentes")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0)
    
    # Herramientas básicas
    tools = [
        Tool(
            name="Calculadora",
            func=lambda x: str(eval(x)),
            description="Útil para realizar cálculos matemáticos"
        ),
        Tool(
            name="Buscador",
            func=lambda x: f"Resultado de búsqueda para: {x}",
            description="Busca información en la web"
        )
    ]
    
    # 1. Agente Zero-shot
    print("\n🔹 Agente Zero-shot:")
    agent_zero_shot = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    start_time = time.time()
    result = agent_zero_shot.run("Calcula 15 * 23 y luego busca información sobre Python")
    elapsed_time = time.time() - start_time
    
    print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    print(f"📊 Resultado: {result}")
    
    return agent_zero_shot

def agentes_con_herramientas_personalizadas():
    """Agentes con herramientas personalizadas"""
    print_step("Herramientas Personalizadas", "Creando agentes con herramientas especializadas")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0)
    
    # Herramientas personalizadas
    tools = [
        calcular_edad,
        convertir_moneda,
        analizar_texto,
        CalculadoraAvanzada(),
        BuscadorWeb()
    ]
    
    # Crear agente
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Casos de prueba
    casos_prueba = [
        "Calcula la edad de alguien nacido el 1990-05-15",
        "Convierte 100 USD a EUR usando la herramienta de conversión",
        "Analiza el texto: 'La inteligencia artificial está transformando el mundo'",
        "Calcula la raíz cuadrada de 144 usando la calculadora avanzada",
        "Busca información sobre LangChain"
    ]
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\n🧪 Caso {i}: {caso}")
        start_time = time.time()
        try:
            result = agent.run(caso)
            print(f"✅ Resultado: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return agent

def agentes_con_memoria():
    """Agentes que mantienen contexto de conversación"""
    print_step("Agentes con Memoria", "Implementando memoria en agentes")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    
    # Herramientas
    tools = [
        Tool(
            name="Calculadora",
            func=lambda x: str(eval(x)),
            description="Para cálculos matemáticos"
        ),
        Tool(
            name="Buscador",
            func=lambda x: f"Información sobre: {x}",
            description="Para buscar información"
        )
    ]
    
    # Memoria de conversación
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Agente con memoria
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    # Conversación de prueba
    conversacion = [
        "Hola, mi nombre es María",
        "Tengo 25 años",
        "¿Recuerdas mi nombre?",
        "Calcula mi edad en días",
        "¿Qué edad tengo en días?"
    ]
    
    print("💬 Iniciando conversación con agente con memoria...")
    
    for mensaje in conversacion:
        print(f"\n👤 Usuario: {mensaje}")
        start_time = time.time()
        try:
            respuesta = agent.run(mensaje)
            print(f"🤖 Agente: {respuesta}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return agent

def agentes_especializados():
    """Agentes diseñados para tareas específicas"""
    print_step("Agentes Especializados", "Creando agentes para tareas específicas")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.3)
    
    # 1. Agente de Análisis Financiero
    def analizar_finanzas(texto: str) -> str:
        """Analiza información financiera"""
        return f"Análisis financiero de: {texto[:100]}..."
    
    def calcular_indicadores(datos: str) -> str:
        """Calcula indicadores financieros"""
        return "ROE: 15%, ROA: 8%, Margen: 12%"
    
    tools_financieras = [
        Tool(name="analizar_finanzas", func=analizar_finanzas, 
             description="Analiza información financiera"),
        Tool(name="calcular_indicadores", func=calcular_indicadores,
             description="Calcula indicadores financieros")
    ]
    
    agente_financiero = initialize_agent(
        tools_financieras,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("💰 Probando Agente Financiero...")
    start_time = time.time()
    resultado_financiero = agente_financiero.run(
        "Analiza los datos financieros de Apple Inc. y calcula sus indicadores principales"
    )
    elapsed_time = time.time() - start_time
    print(f"✅ Resultado: {resultado_financiero}")
    print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    # 2. Agente de Análisis de Texto
    def extraer_entidades(texto: str) -> str:
        """Extrae entidades nombradas"""
        return "Entidades: Apple, Tim Cook, California"
    
    def analizar_sentimiento(texto: str) -> str:
        """Analiza el sentimiento del texto"""
        return "Sentimiento: Positivo (85% confianza)"
    
    tools_texto = [
        Tool(name="extraer_entidades", func=extraer_entidades,
             description="Extrae entidades nombradas del texto"),
        Tool(name="analizar_sentimiento", func=analizar_sentimiento,
             description="Analiza el sentimiento del texto")
    ]
    
    agente_texto = initialize_agent(
        tools_texto,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("\n📝 Probando Agente de Análisis de Texto...")
    start_time = time.time()
    resultado_texto = agente_texto.run(
        "Analiza el siguiente texto: 'Apple lanzó el nuevo iPhone con excelentes características'"
    )
    elapsed_time = time.time() - start_time
    print(f"✅ Resultado: {resultado_texto}")
    print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return {
        'financiero': agente_financiero,
        'texto': agente_texto
    }

def debugging_monitoreo():
    """Debugging y monitoreo de agentes"""
    print_step("Debugging y Monitoreo", "Implementando monitoreo avanzado")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0)
    
    # Callback personalizado
    callback_handler = AgentCallbackHandler()
    
    # Herramientas
    tools = [
        Tool(name="Calculadora", func=lambda x: str(eval(x)),
             description="Para cálculos matemáticos"),
        Tool(name="Buscador", func=lambda x: f"Info: {x}",
             description="Para buscar información")
    ]
    
    # Agente con callback
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        callbacks=[callback_handler]
    )
    
    # Métricas
    metrics = AgentMetrics()
    
    # Casos de prueba con monitoreo
    casos = [
        "Calcula 25 * 4 + 10",
        "Busca información sobre machine learning",
        "Calcula la raíz cuadrada de 100"
    ]
    
    print("🔍 Ejecutando casos con monitoreo...")
    
    for caso in casos:
        print(f"\n🧪 Probando: {caso}")
        
        with get_openai_callback() as cb:
            start_time = time.time()
            try:
                result = agent.run(caso)
                success = True
                error_msg = None
            except Exception as e:
                result = str(e)
                success = False
                error_msg = str(e)
            elapsed_time = time.time() - start_time
        
        # Crear resultado
        agent_result = AgentResult(
            success=success,
            output=result,
            steps_taken=len(callback_handler.steps),
            execution_time=elapsed_time,
            tools_used=[step['tool'] for step in callback_handler.steps],
            error_message=error_msg
        )
        
        # Actualizar métricas
        metrics.update_metrics(agent_result)
        
        print(f"✅ Éxito: {success}")
        print(f"📊 Pasos: {agent_result.steps_taken}")
        print(f"⏱️ Tiempo: {agent_result.execution_time:.3f}s")
        print(f"🔧 Herramientas: {agent_result.tools_used}")
        print(f"💬 Tokens usados: {cb.total_tokens}")
        print(f"💰 Costo: ${cb.total_cost:.4f}")
    
    # Mostrar métricas finales
    print(f"\n📈 Métricas Finales:")
    print(f"   Total de llamadas: {metrics.total_calls}")
    print(f"   Llamadas exitosas: {metrics.successful_calls}")
    print(f"   Llamadas fallidas: {metrics.failed_calls}")
    print(f"   Tiempo promedio: {metrics.average_execution_time:.3f}s")
    print(f"   Uso de herramientas: {metrics.tools_usage}")
    
    return {
        'agent': agent,
        'metrics': metrics,
        'callback_handler': callback_handler
    }

def agentes_multi_herramienta():
    """Agentes que combinan múltiples herramientas"""
    print_step("Agentes Multi-Herramienta", "Combinando múltiples herramientas")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.3)
    
    # Herramientas especializadas
    tools = [
        # Herramientas de cálculo
        Tool(name="Calculadora", func=lambda x: str(eval(x)),
             description="Para cálculos matemáticos básicos"),
        CalculadoraAvanzada(),
        
        # Herramientas de análisis
        analizar_texto,
        Tool(name="Contador_palabras", func=lambda x: f"Palabras: {len(x.split())}",
             description="Cuenta palabras en un texto"),
        
        # Herramientas de conversión
        convertir_moneda,
        Tool(name="Convertir_temperatura", 
             func=lambda x: f"Conversión: {x}",
             description="Convierte temperaturas"),
        
        # Herramientas de búsqueda
        BuscadorWeb(),
        Tool(name="Traductor", func=lambda x: f"Traducción: {x}",
             description="Traduce texto")
    ]
    
    # Agente multi-herramienta
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5
    )
    
    # Casos complejos que requieren múltiples herramientas
    casos_complejos = [
        "Analiza el texto 'Python es excelente para machine learning' y cuenta las palabras",
        "Convierte 50 USD a EUR usando la herramienta de conversión y luego calcula cuánto sería 10% de esa cantidad",
        "Busca información sobre inteligencia artificial y analiza el texto resultante",
        "Calcula la raíz cuadrada de 225 y convierte el resultado a temperatura Celsius"
    ]
    
    print("🛠️ Probando agente multi-herramienta...")
    
    for i, caso in enumerate(casos_complejos, 1):
        print(f"\n🧪 Caso {i}: {caso}")
        start_time = time.time()
        try:
            result = agent.run(caso)
            print(f"✅ Resultado: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return agent

def agentes_con_validacion():
    """Agentes con validación de entrada y salida"""
    print_step("Agentes con Validación", "Implementando validación robusta")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0)
    
    # Herramientas con validación
    def calculadora_validada(expresion: str) -> str:
        """Calculadora con validación de entrada"""
        # Validar caracteres permitidos
        caracteres_permitidos = set("0123456789+-*/(). ")
        if not all(c in caracteres_permitidos for c in expresion):
            return "Error: Solo se permiten números y operadores básicos (+, -, *, /, (, ), .)"
        
        try:
            resultado = eval(expresion)
            if not isinstance(resultado, (int, float)):
                return "Error: El resultado debe ser un número"
            return f"Resultado: {resultado}"
        except ZeroDivisionError:
            return "Error: División por cero no permitida"
        except Exception as e:
            return f"Error en el cálculo: {str(e)}"
    
    def buscador_validado(query: str) -> str:
        """Buscador con validación de entrada"""
        if len(query.strip()) < 3:
            return "Error: La búsqueda debe tener al menos 3 caracteres"
        
        if len(query) > 100:
            return "Error: La búsqueda es demasiado larga (máximo 100 caracteres)"
        
        # Simular búsqueda
        return f"Resultados para '{query}': Información relevante encontrada"
    
    tools_validadas = [
        Tool(name="Calculadora_Validada", func=calculadora_validada,
             description="Calculadora con validación de entrada"),
        Tool(name="Buscador_Validado", func=buscador_validado,
             description="Buscador con validación de entrada")
    ]
    
    # Agente con validación
    agent = initialize_agent(
        tools_validadas,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Casos de prueba con validación
    casos_validacion = [
        "Calcula 2 + 2",  # Caso válido
        "Calcula 10 / 0",  # División por cero
        "Calcula 2 + 'texto'",  # Tipo inválido
        "Busca 'python'",  # Búsqueda válida
        "Busca 'a'",  # Búsqueda muy corta
        "Busca " + "x" * 150,  # Búsqueda muy larga
    ]
    
    print("✅ Probando validación de agentes...")
    
    for caso in casos_validacion:
        print(f"\n🧪 Probando: {caso}")
        start_time = time.time()
        try:
            result = agent.run(caso)
            print(f"✅ Resultado: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return agent

async def agentes_asincronos():
    """Agentes que ejecutan operaciones asíncronas"""
    print_step("Agentes Asíncronos", "Implementando operaciones asíncronas")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0)
    
    # Herramientas asíncronas
    async def calculadora_async(expresion: str) -> str:
        """Calculadora asíncrona"""
        await asyncio.sleep(0.1)  # Simular operación asíncrona
        return f"Resultado async: {eval(expresion)}"
    
    async def buscador_async(query: str) -> str:
        """Buscador asíncrono"""
        await asyncio.sleep(0.2)  # Simular búsqueda asíncrona
        return f"Resultado búsqueda async: {query}"
    
    # Crear herramientas asíncronas
    tools_async = [
        Tool(name="Calculadora_Async", func=calculadora_async,
             description="Calculadora asíncrona"),
        Tool(name="Buscador_Async", func=buscador_async,
             description="Buscador asíncrono")
    ]
    
    # Agente asíncrono
    agent = initialize_agent(
        tools_async,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Casos de prueba asíncronos
    casos_async = [
        "Calcula 15 * 3 usando la calculadora asíncrona",
        "Busca 'machine learning' de forma asíncrona",
        "Calcula 100 / 4 y luego busca 'python'"
    ]
    
    print("⚡ Probando agentes asíncronos...")
    
    for caso in casos_async:
        print(f"\n🧪 Probando: {caso}")
        start_time = time.time()
        try:
            result = await agent.arun(caso)
            print(f"✅ Resultado: {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return agent

def casos_uso_practicos():
    """Casos de uso prácticos de agentes"""
    print_step("Casos de Uso Prácticos", "Implementando aplicaciones reales")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.3)
    
    # 1. Agente de Asistente Personal
    def obtener_clima(ciudad: str) -> str:
        """Simula obtener el clima de una ciudad"""
        return f"Clima en {ciudad}: 22°C, soleado"
    
    def recordar_evento(evento: str) -> str:
        """Simula recordar un evento"""
        return f"Evento recordado: {evento}"
    
    def calcular_ruta(origen: str, destino: str) -> str:
        """Simula calcular una ruta"""
        return f"Ruta de {origen} a {destino}: 15 minutos"
    
    tools_asistente = [
        Tool(name="Clima", func=obtener_clima, description="Obtiene el clima de una ciudad"),
        Tool(name="Recordar", func=recordar_evento, description="Recuerda un evento"),
        Tool(name="Ruta", func=calcular_ruta, description="Calcula una ruta"),
        Tool(name="Calculadora", func=lambda x: str(eval(x)), description="Calculadora")
    ]
    
    asistente_personal = initialize_agent(
        tools_asistente,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("👤 Probando Asistente Personal...")
    start_time = time.time()
    resultado_asistente = asistente_personal.run(
        "¿Cuál es el clima en Madrid? Luego recuerda que tengo una reunión mañana a las 10 AM"
    )
    elapsed_time = time.time() - start_time
    print(f"✅ Resultado: {resultado_asistente}")
    print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    # 2. Agente de Análisis de Datos
    def analizar_dataset(datos: str) -> str:
        """Simula análisis de dataset"""
        return "Análisis: 1000 registros, 5 columnas, sin valores faltantes"
    
    def generar_grafico(tipo: str, datos: str) -> str:
        """Simula generación de gráfico"""
        return f"Gráfico {tipo} generado para los datos"
    
    def calcular_estadisticas(datos: str) -> str:
        """Simula cálculo de estadísticas"""
        return "Media: 45.2, Mediana: 42.1, Desv. Est.: 12.3"
    
    tools_datos = [
        Tool(name="Analizar_Dataset", func=analizar_dataset, description="Analiza un dataset"),
        Tool(name="Generar_Grafico", func=generar_grafico, description="Genera gráficos"),
        Tool(name="Estadisticas", func=calcular_estadisticas, description="Calcula estadísticas")
    ]
    
    agente_datos = initialize_agent(
        tools_datos,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    print("\n📊 Probando Agente de Análisis de Datos...")
    start_time = time.time()
    resultado_datos = agente_datos.run(
        "Analiza el dataset de ventas y genera un gráfico de barras con las estadísticas"
    )
    elapsed_time = time.time() - start_time
    print(f"✅ Resultado: {resultado_datos}")
    print(f"⏱️ Tiempo: {elapsed_time:.3f}s")
    
    return {
        'asistente_personal': asistente_personal,
        'agente_datos': agente_datos
    }

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del módulo"""
    print_separator("MÓDULO 6: AGENTES BÁSICOS")
    
    # Validar configuración
    try:
        config.validate_config()
        print("✅ Configuración validada correctamente")
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return
    
    # Ejecutar todos los ejemplos
    try:
        # 1. Agentes básicos
        agentes_basicos()
        
        # 2. Herramientas personalizadas
        agentes_con_herramientas_personalizadas()
        
        # 3. Agentes con memoria
        agentes_con_memoria()
        
        # 4. Agentes especializados
        agentes_especializados()
        
        # 5. Debugging y monitoreo
        debugging_monitoreo()
        
        # 6. Agentes multi-herramienta
        agentes_multi_herramienta()
        
        # 7. Agentes con validación
        agentes_con_validacion()
        
        # 8. Agentes asíncronos
        asyncio.run(agentes_asincronos())
        
        # 9. Casos de uso prácticos
        casos_uso_practicos()
        
        print_separator("COMPLETADO")
        print("🎉 ¡Módulo 6 completado! Ahora dominas los agentes básicos.")
        print("🚀 Próximo módulo: Integraciones Externas")
        
    except Exception as e:
        print(f"❌ Error en el módulo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
