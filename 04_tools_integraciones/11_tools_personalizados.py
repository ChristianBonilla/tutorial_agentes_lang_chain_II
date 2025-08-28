#!/usr/bin/env python3
"""
============================================================
    MÓDULO 11: TOOLS PERSONALIZADAS AVANZADAS
============================================================

🎯 OBJETIVOS:
- Crear tools complejas y especializadas
- Implementar validación avanzada
- Manejo de errores robusto
- Tools con estado y memoria
- Integración con APIs externas
- Tools asíncronas

📚 CONTENIDO:
1. Tools complejas con estado
2. Validación avanzada
3. Manejo de errores robusto
4. Tools asíncronas
5. Integración con APIs
6. Tools con memoria
7. Herramientas especializadas
"""

import asyncio
import time
import json
import requests
import sqlite3
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator

# Importaciones locales
import sys
sys.path.append('..')
from utils.config import LangChainConfig
from utils.helpers import print_separator, print_step, print_result

# Configuración
config = LangChainConfig()

class WeatherInput(BaseModel):
    """Input para tool del clima"""
    city: str = Field(description="Ciudad para obtener el clima")
    country: Optional[str] = Field(default="", description="País (opcional)")

class DatabaseInput(BaseModel):
    """Input para tool de base de datos"""
    query: str = Field(description="Query SQL a ejecutar")
    operation: str = Field(description="Tipo de operación: SELECT, INSERT, UPDATE, DELETE")

class FileInput(BaseModel):
    """Input para tool de archivos"""
    file_path: str = Field(description="Ruta del archivo")
    operation: str = Field(description="Operación: READ, WRITE, APPEND, DELETE")

class AdvancedWeatherTool(BaseTool):
    """Tool avanzada para obtener información del clima"""
    
    name = "advanced_weather"
    description = "Obtiene información detallada del clima para una ciudad específica"
    args_schema = WeatherInput
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.cache = {}
        self.cache_duration = 300  # 5 minutos
    
    def _run(self, city: str, country: str = "") -> str:
        """Ejecuta la tool"""
        try:
            # Verificar cache
            cache_key = f"{city}_{country}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['data']
            
            # Simular API call (en producción usarías la API real)
            weather_data = self._get_weather_data(city, country)
            
            # Guardar en cache
            self.cache[cache_key] = {
                'data': weather_data,
                'timestamp': time.time()
            }
            
            return weather_data
            
        except Exception as e:
            return f"Error obteniendo clima para {city}: {str(e)}"
    
    def _get_weather_data(self, city: str, country: str) -> str:
        """Simula obtención de datos del clima"""
        # En producción, aquí harías la llamada real a la API
        import random
        
        temperatures = random.randint(15, 35)
        conditions = ["Soleado", "Nublado", "Lluvioso", "Parcialmente nublado"]
        condition = random.choice(conditions)
        
        return f"Clima en {city}, {country if country else 'País no especificado'}: {temperatures}°C, {condition}"

class DatabaseTool(BaseTool):
    """Tool para operaciones de base de datos"""
    
    name = "database_operations"
    description = "Ejecuta operaciones SQL en una base de datos SQLite"
    args_schema = DatabaseInput
    
    def __init__(self, db_path: str = "data/tools_database.db"):
        super().__init__()
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Crear tabla de ejemplo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insertar datos de ejemplo
        cursor.execute('''
            INSERT OR IGNORE INTO users (name, email) VALUES 
            ('Juan Pérez', 'juan@example.com'),
            ('María García', 'maria@example.com'),
            ('Carlos López', 'carlos@example.com')
        ''')
        
        conn.commit()
        conn.close()
    
    def _run(self, query: str, operation: str) -> str:
        """Ejecuta la operación de base de datos"""
        try:
            # Validar operación
            operation = operation.upper()
            if operation not in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
                return f"Operación no válida: {operation}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ejecutar query
            cursor.execute(query)
            
            if operation == 'SELECT':
                results = cursor.fetchall()
                conn.close()
                return f"Resultados: {results}"
            else:
                conn.commit()
                conn.close()
                return f"Operación {operation} ejecutada exitosamente"
                
        except Exception as e:
            return f"Error en operación de base de datos: {str(e)}"

class FileOperationsTool(BaseTool):
    """Tool para operaciones de archivos"""
    
    name = "file_operations"
    description = "Realiza operaciones de lectura, escritura y manipulación de archivos"
    args_schema = FileInput
    
    def __init__(self, base_path: str = "data/files"):
        super().__init__()
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def _run(self, file_path: str, operation: str) -> str:
        """Ejecuta la operación de archivo"""
        try:
            full_path = os.path.join(self.base_path, file_path)
            operation = operation.upper()
            
            if operation == 'READ':
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"Contenido del archivo {file_path}:\n{content}"
                else:
                    return f"Archivo {file_path} no encontrado"
            
            elif operation == 'WRITE':
                # Para escritura, necesitaríamos contenido adicional
                return f"Operación WRITE requiere contenido adicional"
            
            elif operation == 'APPEND':
                # Para append, necesitaríamos contenido adicional
                return f"Operación APPEND requiere contenido adicional"
            
            elif operation == 'DELETE':
                if os.path.exists(full_path):
                    os.remove(full_path)
                    return f"Archivo {file_path} eliminado exitosamente"
                else:
                    return f"Archivo {file_path} no encontrado"
            
            else:
                return f"Operación no válida: {operation}"
                
        except Exception as e:
            return f"Error en operación de archivo: {str(e)}"

class AsyncAPITool(BaseTool):
    """Tool asíncrona para llamadas a APIs"""
    
    name = "async_api_call"
    description = "Realiza llamadas asíncronas a APIs externas"
    
    def __init__(self):
        super().__init__()
        self.session = None
    
    async def _arun(self, url: str, method: str = "GET", data: str = "") -> str:
        """Ejecuta la llamada asíncrona a la API"""
        try:
            import aiohttp
            
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            method = method.upper()
            
            if method == "GET":
                async with self.session.get(url) as response:
                    return await response.text()
            
            elif method == "POST":
                async with self.session.post(url, data=data) as response:
                    return await response.text()
            
            else:
                return f"Método HTTP no soportado: {method}"
                
        except Exception as e:
            return f"Error en llamada API: {str(e)}"

class StatefulTool(BaseTool):
    """Tool con estado y memoria"""
    
    name = "stateful_operations"
    description = "Tool que mantiene estado entre llamadas"
    
    def __init__(self):
        super().__init__()
        self.state = {
            'counter': 0,
            'history': [],
            'last_operation': None
        }
    
    def _run(self, operation: str, value: str = "") -> str:
        """Ejecuta operación con estado"""
        try:
            operation = operation.lower()
            
            if operation == 'increment':
                self.state['counter'] += 1
                self.state['history'].append(f"Increment at {datetime.now()}")
                self.state['last_operation'] = 'increment'
                return f"Contador incrementado. Valor actual: {self.state['counter']}"
            
            elif operation == 'decrement':
                self.state['counter'] -= 1
                self.state['history'].append(f"Decrement at {datetime.now()}")
                self.state['last_operation'] = 'decrement'
                return f"Contador decrementado. Valor actual: {self.state['counter']}"
            
            elif operation == 'get_state':
                return f"Estado actual: {json.dumps(self.state, default=str)}"
            
            elif operation == 'reset':
                self.state = {
                    'counter': 0,
                    'history': [],
                    'last_operation': 'reset'
                }
                return "Estado reseteado"
            
            else:
                return f"Operación no válida: {operation}"
                
        except Exception as e:
            return f"Error en operación con estado: {str(e)}"

class ValidationTool(BaseTool):
    """Tool con validación avanzada"""
    
    name = "validation_tool"
    description = "Tool que valida datos con reglas complejas"
    
    class ValidationInput(BaseModel):
        email: str = Field(description="Email a validar")
        age: int = Field(description="Edad a validar")
        password: str = Field(description="Contraseña a validar")
        
        @validator('email')
        def validate_email(cls, v):
            if '@' not in v or '.' not in v:
                raise ValueError('Email inválido')
            return v
        
        @validator('age')
        def validate_age(cls, v):
            if v < 0 or v > 150:
                raise ValueError('Edad debe estar entre 0 y 150')
            return v
        
        @validator('password')
        def validate_password(cls, v):
            if len(v) < 8:
                raise ValueError('Contraseña debe tener al menos 8 caracteres')
            if not any(c.isupper() for c in v):
                raise ValueError('Contraseña debe contener al menos una mayúscula')
            if not any(c.isdigit() for c in v):
                raise ValueError('Contraseña debe contener al menos un número')
            return v
    
    args_schema = ValidationInput
    
    def _run(self, email: str, age: int, password: str) -> str:
        """Valida los datos proporcionados"""
        try:
            # La validación ya se hace automáticamente por Pydantic
            return f"✅ Validación exitosa:\nEmail: {email}\nEdad: {age}\nContraseña: {'*' * len(password)}"
        except Exception as e:
            return f"❌ Error de validación: {str(e)}"

def crear_tools_avanzadas():
    """Crea y configura tools avanzadas"""
    print_step(1, "Tools Avanzadas", "Creando herramientas especializadas")
    
    tools = []
    
    # Weather Tool
    weather_tool = AdvancedWeatherTool()
    tools.append(weather_tool)
    
    # Database Tool
    db_tool = DatabaseTool()
    tools.append(db_tool)
    
    # File Operations Tool
    file_tool = FileOperationsTool()
    tools.append(file_tool)
    
    # Stateful Tool
    stateful_tool = StatefulTool()
    tools.append(stateful_tool)
    
    # Validation Tool
    validation_tool = ValidationTool()
    tools.append(validation_tool)
    
    print(f"✅ {len(tools)} tools avanzadas creadas")
    return tools

def probar_tools_individuales():
    """Prueba cada tool individualmente"""
    print_step(2, "Prueba Individual", "Probando cada tool por separado")
    
    tools = crear_tools_avanzadas()
    
    # Probar Weather Tool
    print("\n🌤️ Probando Weather Tool:")
    weather_result = tools[0].run("Madrid", "España")
    print(f"Resultado: {weather_result}")
    
    # Probar Database Tool
    print("\n🗄️ Probando Database Tool:")
    db_result = tools[1].run("SELECT * FROM users", "SELECT")
    print(f"Resultado: {db_result}")
    
    # Probar Stateful Tool
    print("\n🧠 Probando Stateful Tool:")
    stateful_result1 = tools[3].run("increment")
    print(f"Resultado 1: {stateful_result1}")
    stateful_result2 = tools[3].run("get_state")
    print(f"Resultado 2: {stateful_result2}")
    
    # Probar Validation Tool
    print("\n✅ Probando Validation Tool:")
    try:
        validation_result = tools[4].run("test@example.com", 25, "Password123")
        print(f"Resultado: {validation_result}")
    except Exception as e:
        print(f"Error de validación: {e}")

def crear_agente_con_tools():
    """Crea un agente que usa las tools avanzadas"""
    print_step(3, "Agente con Tools", "Creando agente que usa tools avanzadas")
    
    llm = ChatOpenAI(model=config.default_model, temperature=0.7)
    tools = crear_tools_avanzadas()
    
    # Crear agente
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Probar agente
    test_queries = [
        "¿Cuál es el clima en Barcelona?",
        "Muestra todos los usuarios en la base de datos",
        "Incrementa el contador y luego muéstrame el estado",
        "Valida el email 'usuario@test.com' con edad 30 y contraseña 'TestPass123'"
    ]
    
    print("\n🤖 Probando agente con tools avanzadas:")
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            result = agent.run(query)
            print(f"🤖 Respuesta: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

def tools_asincronas():
    """Demuestra el uso de tools asíncronas"""
    print_step(4, "Tools Asíncronas", "Implementando tools asíncronas")
    
    async def test_async_tools():
        # Crear tool asíncrona
        async_tool = AsyncAPITool()
        
        # URLs de prueba (APIs públicas)
        test_urls = [
            "https://jsonplaceholder.typicode.com/posts/1",
            "https://jsonplaceholder.typicode.com/users/1"
        ]
        
        print("🔄 Probando tools asíncronas:")
        
        for url in test_urls:
            print(f"\n📡 Llamando a: {url}")
            result = await async_tool._arun(url)
            print(f"📄 Respuesta: {result[:100]}...")
    
    # Ejecutar prueba asíncrona
    asyncio.run(test_async_tools())

def tools_con_manejo_errores():
    """Demuestra manejo robusto de errores en tools"""
    print_step(5, "Manejo de Errores", "Implementando manejo robusto de errores")
    
    class RobustTool(BaseTool):
        """Tool con manejo robusto de errores"""
        
        name = "robust_tool"
        description = "Tool que maneja errores de forma robusta"
        
        def _run(self, operation: str) -> str:
            try:
                if operation == "success":
                    return "Operación exitosa"
                
                elif operation == "division_by_zero":
                    result = 1 / 0
                    return str(result)
                
                elif operation == "file_not_found":
                    with open("archivo_inexistente.txt", "r") as f:
                        return f.read()
                
                elif operation == "network_error":
                    response = requests.get("https://api.inexistente.com", timeout=1)
                    return response.text
                
                else:
                    return f"Operación desconocida: {operation}"
                    
            except ZeroDivisionError:
                return "Error: División por cero detectada y manejada"
            except FileNotFoundError:
                return "Error: Archivo no encontrado, manejado correctamente"
            except requests.RequestException:
                return "Error: Problema de red manejado correctamente"
            except Exception as e:
                return f"Error general manejado: {str(e)}"
    
    # Probar tool robusta
    robust_tool = RobustTool()
    
    test_operations = ["success", "division_by_zero", "file_not_found", "network_error"]
    
    print("🛡️ Probando manejo robusto de errores:")
    for operation in test_operations:
        print(f"\n🔧 Operación: {operation}")
        result = robust_tool.run(operation)
        print(f"📋 Resultado: {result}")

def tools_especializadas():
    """Crea tools especializadas para casos de uso específicos"""
    print_step(6, "Tools Especializadas", "Creando tools para casos de uso específicos")
    
    class DataAnalysisTool(BaseTool):
        """Tool para análisis de datos"""
        
        name = "data_analysis"
        description = "Analiza datos y genera estadísticas"
        
        def _run(self, data: str) -> str:
            try:
                # Parsear datos (formato: "1,2,3,4,5")
                numbers = [float(x.strip()) for x in data.split(',')]
                
                if not numbers:
                    return "No hay datos para analizar"
                
                # Calcular estadísticas
                import statistics
                
                stats = {
                    'count': len(numbers),
                    'sum': sum(numbers),
                    'mean': statistics.mean(numbers),
                    'median': statistics.median(numbers),
                    'min': min(numbers),
                    'max': max(numbers)
                }
                
                return f"Análisis estadístico:\n" + "\n".join([f"{k}: {v}" for k, v in stats.items()])
                
            except Exception as e:
                return f"Error en análisis de datos: {str(e)}"
    
    class TextProcessingTool(BaseTool):
        """Tool para procesamiento de texto"""
        
        name = "text_processing"
        description = "Procesa texto y extrae información"
        
        def _run(self, text: str, operation: str) -> str:
            try:
                operation = operation.lower()
                
                if operation == "word_count":
                    words = text.split()
                    return f"Número de palabras: {len(words)}"
                
                elif operation == "character_count":
                    return f"Número de caracteres: {len(text)}"
                
                elif operation == "uppercase":
                    return text.upper()
                
                elif operation == "lowercase":
                    return text.lower()
                
                elif operation == "reverse":
                    return text[::-1]
                
                else:
                    return f"Operación no válida: {operation}"
                    
            except Exception as e:
                return f"Error en procesamiento de texto: {str(e)}"
    
    # Probar tools especializadas
    data_tool = DataAnalysisTool()
    text_tool = TextProcessingTool()
    
    print("📊 Probando Data Analysis Tool:")
    data_result = data_tool.run("1,2,3,4,5,6,7,8,9,10")
    print(f"Resultado: {data_result}")
    
    print("\n📝 Probando Text Processing Tool:")
    text_result = text_tool.run("Hola mundo, esto es una prueba", "word_count")
    print(f"Resultado: {text_result}")

def main():
    """Función principal"""
    print_separator()
    print("             MÓDULO 11: TOOLS PERSONALIZADAS AVANZADAS             ")
    print_separator()
    
    # Validar configuración
    if not config.validate_config():
        print("❌ Error en configuración")
        return
    
    print("✅ Configuración validada correctamente")
    
    # Ejecutar todos los ejemplos
    probar_tools_individuales()
    crear_agente_con_tools()
    tools_asincronas()
    tools_con_manejo_errores()
    tools_especializadas()
    
    print_separator()
    print("🎉 ¡Módulo 11 completado! Ahora dominas tools personalizadas avanzadas.")
    print("🚀 Próximo módulo: Integraciones Externas")

if __name__ == "__main__":
    main()

