"""
🛠️ MÓDULO 10: TOOLS BÁSICOS
============================

En este módulo aprenderás a usar y crear herramientas (tools) en LangChain
que permiten a los LLMs interactuar con el mundo exterior y realizar tareas específicas.

🎯 OBJETIVOS DE APRENDIZAJE:
- Entender qué son las tools y cómo funcionan
- Usar tools predefinidas de LangChain
- Crear tools personalizadas
- Integrar tools con agents
- Manejar errores en tools

📚 CONCEPTOS CLAVE:
- Tool Definition
- Tool Execution
- Tool Integration
- Custom Tools
- Tool Validation
- Error Handling

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os
import requests
import json
from typing import List, Dict, Any
from datetime import datetime

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from utils.helpers import print_separator, print_step, print_result

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain_community.tools import Tool, DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.agents import tool
from langchain_community.utilities import WikipediaAPIWrapper

def introduccion_tools():
    """
    🎓 Introducción a las Tools
    """
    print_separator("¿QUÉ SON LAS TOOLS?")
    
    print("""
    🛠️ Las Tools son funciones que permiten a los LLMs interactuar con
    sistemas externos y realizar tareas específicas que van más allá
    del procesamiento de texto.
    
    🏗️ ARQUITECTURA DE TOOLS:
    
    [User Input] → [Agent] → [Tool Selection] → [Tool Execution] → [Result] → [Response]
    
    📋 TIPOS DE TOOLS:
    
    1. 🔍 Search Tools: Búsqueda en internet
    2. 📊 Data Tools: Manipulación de datos
    3. 🌐 API Tools: Integración con APIs externas
    4. 📁 File Tools: Manejo de archivos
    5. 🧮 Math Tools: Cálculos matemáticos
    6. 🎯 Custom Tools: Herramientas personalizadas
    
    💡 VENTAJAS:
    - Acceso a información actualizada
    - Capacidades específicas de dominio
    - Integración con sistemas externos
    - Automatización de tareas complejas
    """)

def tools_predefinidas():
    """
    🔧 Tools predefinidas de LangChain
    """
    print_step(1, "Tools Predefinidas", "Usando herramientas incorporadas")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Tool de búsqueda
        search_tool = DuckDuckGoSearchRun()
        
        # Tool de Wikipedia
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        # Tool personalizada simple
        @tool
        def get_current_time():
            """Obtiene la hora actual"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tool personalizada con parámetros
        @tool
        def calculate_math(expression: str) -> str:
            """Evalúa una expresión matemática simple"""
            try:
                # Solo permitir operaciones matemáticas básicas por seguridad
                allowed_chars = set('0123456789+-*/(). ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Solo se permiten operaciones matemáticas básicas"
                
                result = eval(expression)
                return f"Resultado: {result}"
            except Exception as e:
                return f"Error en el cálculo: {str(e)}"
        
        tools = [search_tool, wikipedia_tool, get_current_time, calculate_math]
        
        print_result("Tools Creadas", tools)
        
        # Crear agent con tools
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        print_result("Agent con Tools", agent)
        
        # Probar tools individualmente
        print("\n🔧 Probando tools individualmente:")
        
        # Probar búsqueda
        print(f"\n🔍 Búsqueda: {search_tool.run('Python programming language')[:100]}...")
        
        # Probar Wikipedia
        print(f"\n📚 Wikipedia: {wikipedia_tool.run('Artificial Intelligence')[:100]}...")
        
        # Probar hora actual
        print(f"\n⏰ Hora actual: {get_current_time()}")
        
        # Probar cálculo
        print(f"\n🧮 Cálculo: {calculate_math('2 + 3 * 4')}")
        
        return agent, tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def custom_tools_basicas():
    """
    🎯 Creando tools personalizadas básicas
    """
    print_step(2, "Custom Tools Básicas", "Creando herramientas personalizadas")
    
    try:
        # Tool personalizada usando decorador
        @tool
        def get_weather(city: str) -> str:
            """Obtiene información del clima para una ciudad (simulado)"""
            # Simulación - en un caso real usarías una API real
            weather_data = {
                "Madrid": "Soleado, 25°C",
                "Barcelona": "Nublado, 22°C",
                "Valencia": "Lluvioso, 18°C",
                "Sevilla": "Soleado, 30°C"
            }
            
            if city in weather_data:
                return f"Clima en {city}: {weather_data[city]}"
            else:
                return f"No tengo información del clima para {city}"
        
        # Tool personalizada usando clase
        class TranslationTool(BaseTool):
            name: str = "translate_text"
            description: str = "Traduce texto entre idiomas"
            
            def _run(self, text: str, from_lang: str = "es", to_lang: str = "en") -> str:
                # Simulación de traducción
                translations = {
                    ("hola", "es", "en"): "hello",
                    ("hello", "en", "es"): "hola",
                    ("gracias", "es", "en"): "thank you",
                    ("thank you", "en", "es"): "gracias"
                }
                
                key = (text.lower(), from_lang, to_lang)
                if key in translations:
                    return translations[key]
                else:
                    return f"Traducción no disponible para: {text} ({from_lang} → {to_lang})"
            
            def _arun(self, text: str, from_lang: str = "es", to_lang: str = "en"):
                # Implementación asíncrona (opcional)
                return self._run(text, from_lang, to_lang)
        
        # Tool para análisis de texto
        @tool
        def analyze_text(text: str) -> str:
            """Analiza un texto y proporciona estadísticas básicas"""
            if not text:
                return "Error: Texto vacío"
            
            words = text.split()
            chars = len(text)
            sentences = text.count('.') + text.count('!') + text.count('?')
            
            return f"""
            Análisis del texto:
            - Palabras: {len(words)}
            - Caracteres: {chars}
            - Oraciones: {sentences}
            - Palabra más larga: {max(words, key=len) if words else 'N/A'}
            """
        
        # Tool para validación de email
        @tool
        def validate_email(email: str) -> str:
            """Valida si un email tiene formato correcto"""
            import re
            
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if re.match(pattern, email):
                return f"✅ Email válido: {email}"
            else:
                return f"❌ Email inválido: {email}"
        
        tools = [get_weather, TranslationTool(), analyze_text, validate_email]
        
        print_result("Custom Tools Creadas", tools)
        
        # Probar tools
        print("\n🎯 Probando custom tools:")
        
        print(f"\n🌤️ Clima: {get_weather.invoke({'city': 'Madrid'})}")
        print(f"\n🌐 Traducción: {TranslationTool().invoke({'text': 'hola', 'from_lang': 'es', 'to_lang': 'en'})}")
        print(f"\n📊 Análisis: {analyze_text.invoke({'text': 'Hola mundo. Este es un texto de prueba!'})}")
        print(f"\n📧 Validación: {validate_email.invoke({'email': 'usuario@ejemplo.com'})}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def tools_con_parametros():
    """
    🔧 Tools con parámetros complejos
    """
    print_step(3, "Tools con Parámetros", "Creando herramientas más complejas")
    
    try:
        # Tool con múltiples parámetros
        @tool
        def format_text(text: str, format_type: str = "uppercase", max_length: int = 100) -> str:
            """Formatea texto según el tipo especificado"""
            if not text:
                return "Error: Texto vacío"
            
            # Aplicar formato
            if format_type == "uppercase":
                formatted = text.upper()
            elif format_type == "lowercase":
                formatted = text.lower()
            elif format_type == "title":
                formatted = text.title()
            elif format_type == "reverse":
                formatted = text[::-1]
            else:
                return f"Error: Tipo de formato '{format_type}' no soportado"
            
            # Aplicar límite de longitud
            if len(formatted) > max_length:
                formatted = formatted[:max_length] + "..."
            
            return formatted
        
        # Tool para generación de contraseñas
        @tool
        def generate_password(length: int = 8, include_symbols: bool = True, include_numbers: bool = True) -> str:
            """Genera una contraseña segura"""
            import random
            import string
            
            if length < 4:
                return "Error: La longitud mínima es 4 caracteres"
            
            chars = string.ascii_letters
            
            if include_numbers:
                chars += string.digits
            
            if include_symbols:
                chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
            
            password = ''.join(random.choice(chars) for _ in range(length))
            return f"Contraseña generada: {password}"
        
        # Tool para cálculo de fechas
        @tool
        def date_calculator(operation: str, days: int, base_date: str = "today") -> str:
            """Realiza cálculos con fechas"""
            from datetime import datetime, timedelta
            
            try:
                if base_date == "today":
                    base = datetime.now()
                else:
                    base = datetime.strptime(base_date, "%Y-%m-%d")
                
                if operation == "add":
                    result = base + timedelta(days=days)
                elif operation == "subtract":
                    result = base - timedelta(days=days)
                else:
                    return "Error: Operación debe ser 'add' o 'subtract'"
                
                return f"Fecha resultante: {result.strftime('%Y-%m-%d')}"
                
            except Exception as e:
                return f"Error en el cálculo de fecha: {str(e)}"
        
        tools = [format_text, generate_password, date_calculator]
        
        print_result("Tools con Parámetros", tools)
        
        # Probar tools
        print("\n🔧 Probando tools con parámetros:")
        
        print(f"\n📝 Formato: {format_text.invoke({'text': 'hola mundo', 'format_type': 'uppercase', 'max_length': 10})}")
        print(f"\n🔐 Contraseña: {generate_password.invoke({'length': 12, 'include_numbers': True, 'include_symbols': True})}")
        print(f"\n📅 Fecha: {date_calculator.invoke({'operation': 'add', 'days': 7, 'base_date': 'today'})}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def tools_con_validacion():
    """
    ✅ Tools con validación de entrada
    """
    print_step(4, "Tools con Validación", "Validando entradas de usuario")
    
    try:
        # Tool con validación de email
        @tool
        def send_email(to_email: str, subject: str, message: str) -> str:
            """Envía un email (simulado) con validación"""
            import re
            
            # Validar email
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, to_email):
                return "❌ Error: Email inválido"
            
            # Validar asunto
            if not subject or len(subject) < 3:
                return "❌ Error: Asunto debe tener al menos 3 caracteres"
            
            # Validar mensaje
            if not message or len(message) < 10:
                return "❌ Error: Mensaje debe tener al menos 10 caracteres"
            
            # Simular envío
            return f"✅ Email enviado a {to_email}\nAsunto: {subject}\nMensaje: {message[:50]}..."
        
        # Tool con validación de números
        @tool
        def calculate_percentage(value: float, total: float) -> str:
            """Calcula el porcentaje de un valor sobre un total"""
            try:
                # Validar que los valores sean números positivos
                if value < 0 or total < 0:
                    return "❌ Error: Los valores deben ser positivos"
                
                if total == 0:
                    return "❌ Error: El total no puede ser cero"
                
                percentage = (value / total) * 100
                return f"✅ Porcentaje: {percentage:.2f}%"
                
            except Exception as e:
                return f"❌ Error en el cálculo: {str(e)}"
        
        # Tool con validación de URL
        @tool
        def check_website_status(url: str) -> str:
            """Verifica el estado de un sitio web"""
            import re
            
            # Validar formato de URL
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, url):
                return "❌ Error: URL inválida"
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return f"✅ Sitio web {url} está funcionando (Status: {response.status_code})"
                else:
                    return f"⚠️ Sitio web {url} responde con status: {response.status_code}"
                    
            except requests.exceptions.RequestException as e:
                return f"❌ Error al conectar con {url}: {str(e)}"
        
        tools = [send_email, calculate_percentage, check_website_status]
        
        print_result("Tools con Validación", tools)
        
        # Probar tools
        print("\n✅ Probando tools con validación:")
        
        print(f"\n📧 Email: {send_email.invoke({'to_email': 'usuario@ejemplo.com', 'subject': 'Prueba', 'message': 'Este es un mensaje de prueba'})}")
        print(f"\n📊 Porcentaje: {calculate_percentage.invoke({'value': 25, 'total': 100})}")
        print(f"\n🌐 Website: {check_website_status.invoke({'url': 'https://www.google.com'})}")
        
        # Probar casos de error
        print(f"\n❌ Email inválido: {send_email.invoke({'to_email': 'email-invalido', 'subject': 'Test', 'message': 'Mensaje'})}")
        print(f"\n❌ Porcentaje inválido: {calculate_percentage.invoke({'value': -5, 'total': 100})}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def tools_con_errores():
    """
    🛡️ Manejo de errores en tools
    """
    print_step(5, "Manejo de Errores", "Creando tools robustas")
    
    print("""
    🛡️ ESTRATEGIAS DE MANEJO DE ERRORES:
    
    1. 🔄 Try-Catch: Capturar excepciones específicas
    2. ⚠️ Validación: Validar inputs antes de procesar
    3. 🔀 Fallback: Proporcionar alternativas cuando falla
    4. 📊 Logging: Registrar errores para debugging
    5. 🛡️ Timeout: Evitar bloqueos indefinidos
    """)
    
    try:
        # Tool con manejo de errores robusto
        @tool
        def safe_file_operation(operation: str, filename: str, content: str = "") -> str:
            """Operaciones seguras de archivos con manejo de errores"""
            import os
            
            try:
                if operation == "read":
                    if not os.path.exists(filename):
                        return f"❌ Error: Archivo '{filename}' no existe"
                    
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"✅ Contenido del archivo '{filename}': {content[:100]}..."
                
                elif operation == "write":
                    if not filename:
                        return "❌ Error: Nombre de archivo requerido"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return f"✅ Archivo '{filename}' creado exitosamente"
                
                elif operation == "delete":
                    if not os.path.exists(filename):
                        return f"❌ Error: Archivo '{filename}' no existe"
                    
                    os.remove(filename)
                    return f"✅ Archivo '{filename}' eliminado"
                
                else:
                    return f"❌ Error: Operación '{operation}' no soportada"
                    
            except PermissionError:
                return f"❌ Error: Sin permisos para acceder a '{filename}'"
            except Exception as e:
                return f"❌ Error inesperado: {str(e)}"
        
        # Tool con timeout
        @tool
        def api_request_with_timeout(url: str, timeout: int = 5) -> str:
            """Realiza una petición HTTP con timeout"""
            try:
                response = requests.get(url, timeout=timeout)
                return f"✅ Respuesta exitosa: Status {response.status_code}"
                
            except requests.exceptions.Timeout:
                return f"❌ Error: Timeout después de {timeout} segundos"
            except requests.exceptions.ConnectionError:
                return f"❌ Error: No se pudo conectar a {url}"
            except Exception as e:
                return f"❌ Error inesperado: {str(e)}"
        
        # Tool con fallback
        @tool
        def get_currency_rate(currency: str) -> str:
            """Obtiene tasa de cambio de moneda con fallback"""
            # Simulación de API externa
            rates = {
                "USD": 1.0,
                "EUR": 0.85,
                "GBP": 0.73,
                "JPY": 110.0
            }
            
            try:
                if currency.upper() in rates:
                    return f"✅ Tasa de {currency.upper()}: {rates[currency.upper()]}"
                else:
                    # Fallback: usar USD como referencia
                    return f"⚠️ Moneda '{currency}' no encontrada. Usando USD como referencia: 1.0"
                    
            except Exception as e:
                return f"❌ Error al obtener tasa de cambio: {str(e)}"
        
        tools = [safe_file_operation, api_request_with_timeout, get_currency_rate]
        
        print_result("Tools con Manejo de Errores", tools)
        
        # Probar tools
        print("\n🛡️ Probando tools con manejo de errores:")
        
        print(f"\n📁 Archivo: {safe_file_operation.invoke({'operation': 'write', 'filename': 'test.txt', 'content': 'Contenido de prueba'})}")
        print(f"\n🌐 API: {api_request_with_timeout.invoke({'url': 'https://httpbin.org/delay/2', 'timeout': 1})}")
        print(f"\n💰 Moneda: {get_currency_rate.invoke({'currency': 'EUR'})}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def tools_con_agents():
    """
    🤖 Integrando tools con agents
    """
    print_step(6, "Tools con Agents", "Combinando herramientas con agentes")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Tools para el agent
        @tool
        def get_current_time():
            """Obtiene la hora actual"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @tool
        def calculate_math(expression: str) -> str:
            """Evalúa una expresión matemática simple"""
            try:
                allowed_chars = set('0123456789+-*/(). ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Solo operaciones matemáticas básicas"
                
                result = eval(expression)
                return f"Resultado: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def format_text_simple(text: str) -> str:
            """Convierte texto a mayúsculas"""
            return text.upper()
        
        tools = [get_current_time, calculate_math, format_text_simple]
        
        # Crear agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        print_result("Agent con Tools", agent)
        
        # Probar agent con diferentes consultas
        consultas = [
            "¿Qué hora es?",
            "Calcula 15 + 25 * 2",
            "Convierte 'hola mundo' a mayúsculas",
            "¿Cuál es la hora actual?"
        ]
        
        print("\n🤖 Probando agent con tools:")
        for consulta in consultas:
            print(f"\n👤 Usuario: {consulta}")
            try:
                respuesta = agent.run(consulta)
                print(f"🤖 Agent: {respuesta}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def optimizacion_tools():
    """
    ⚡ Optimización de tools
    """
    print_step(7, "Optimización de Tools", "Mejorando el rendimiento")
    
    print("""
    ⚡ ESTRATEGIAS DE OPTIMIZACIÓN:
    
    1. 🔄 Caching: Guardar resultados para reutilizar
    2. ⚡ Async: Usar operaciones asíncronas cuando sea posible
    3. 🎯 Batching: Procesar múltiples items juntos
    4. 📏 Timeout: Evitar bloqueos indefinidos
    5. 🔀 Parallel: Ejecutar tools en paralelo
    """)
    
    try:
        import time
        import asyncio
        
        # Tool con caching simple
        cache = {}
        
        @tool
        def cached_api_call(endpoint: str) -> str:
            """Llamada a API con caching"""
            if endpoint in cache:
                return f"✅ (Cached) Resultado para {endpoint}: {cache[endpoint]}"
            
            # Simular llamada a API
            time.sleep(1)  # Simular delay
            result = f"Datos de {endpoint}"
            cache[endpoint] = result
            
            return f"✅ Resultado para {endpoint}: {result}"
        
        # Tool asíncrona
        @tool
        def async_operation(operation: str) -> str:
            """Operación asíncrona simulada"""
            import asyncio
            
            async def async_task():
                await asyncio.sleep(0.5)  # Simular operación asíncrona
                return f"Operación {operation} completada"
            
            # Ejecutar tarea asíncrona
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(async_task())
            loop.close()
            
            return result
        
        # Tool con timeout
        @tool
        def timeout_operation(operation: str, timeout: int = 3) -> str:
            """Operación con timeout"""
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Operación excedió el tiempo límite")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Simular operación que puede tardar
                time.sleep(2)
                signal.alarm(0)  # Cancelar alarm
                return f"✅ {operation} completada exitosamente"
                
            except TimeoutError:
                return f"❌ {operation} excedió el timeout de {timeout} segundos"
        
        tools = [cached_api_call, async_operation, timeout_operation]
        
        print_result("Tools Optimizadas", tools)
        
        # Probar optimizaciones
        print("\n⚡ Probando optimizaciones:")
        
        # Probar caching
        print(f"\n🔄 Caching: {cached_api_call.invoke({'endpoint': 'users'})}")
        print(f"🔄 Caching (segunda vez): {cached_api_call.invoke({'endpoint': 'users'})}")
        
        # Probar async
        print(f"\n⚡ Async: {async_operation.invoke({'task': 'procesamiento'})}")
        
        # Probar timeout
        print(f"\n⏱️ Timeout: {timeout_operation.invoke({'operation': 'tarea_lenta', 'timeout': 1})}")
        
        return tools
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def ejercicios_practicos():
    """
    🎯 Ejercicios prácticos
    """
    print_step(8, "Ejercicios Prácticos", "Pon en práctica lo aprendido")
    
    ejercicios = [
        {
            "titulo": "Tool de Análisis de Texto",
            "descripcion": "Crea una tool que analice sentimientos de texto",
            "objetivo": "Practicar tools con procesamiento de texto"
        },
        {
            "titulo": "Tool de Validación de Datos",
            "descripcion": "Implementa una tool que valide diferentes tipos de datos",
            "objetivo": "Aprender validación robusta"
        },
        {
            "titulo": "Tool de Integración API",
            "descripcion": "Crea una tool que se conecte a una API externa",
            "objetivo": "Practicar integración con APIs"
        },
        {
            "titulo": "Tool con Caching",
            "descripcion": "Implementa una tool con sistema de cache",
            "objetivo": "Aprender optimización"
        },
        {
            "titulo": "Agent Personalizado",
            "descripcion": "Crea un agent que use múltiples tools personalizadas",
            "objetivo": "Practicar integración completa"
        }
    ]
    
    print("\n🎯 EJERCICIOS PRÁCTICOS:")
    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{i}. {ejercicio['titulo']}")
        print(f"   Descripción: {ejercicio['descripcion']}")
        print(f"   Objetivo: {ejercicio['objetivo']}")

def mejores_practicas():
    """
    ⭐ Mejores prácticas para tools
    """
    print_step(9, "Mejores Prácticas", "Consejos para crear tools efectivas")
    
    practicas = [
        "🛠️ Mantén las tools simples y enfocadas en una tarea específica",
        "✅ Implementa validación robusta de inputs",
        "🛡️ Maneja errores de manera elegante",
        "📝 Documenta claramente el propósito y parámetros de cada tool",
        "⚡ Optimiza el rendimiento con caching y timeouts",
        "🔒 Considera aspectos de seguridad en tools que manejan datos",
        "🧪 Prueba tus tools con diferentes inputs y casos edge",
        "📊 Monitorea el rendimiento y uso de tus tools",
        "🔄 Usa async cuando sea apropiado para operaciones I/O",
        "🎯 Diseña tools que sean reutilizables y modulares"
    ]
    
    print("\n⭐ MEJORES PRÁCTICAS:")
    for practica in practicas:
        print(f"   {practica}")

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 10: TOOLS BÁSICOS")
    
    # Verificar configuración
    if not config.validate_config():
        return
    
    # Contenido del módulo
    introduccion_tools()
    
    # Tools básicas
    tools_predefinidas()
    custom_tools_basicas()
    tools_con_parametros()
    
    # Características avanzadas
    tools_con_validacion()
    tools_con_errores()
    tools_con_agents()
    
    # Optimización
    optimizacion_tools()
    
    # Consolidación
    mejores_practicas()
    ejercicios_practicos()
    
    print("\n🎉 ¡Módulo 10 completado! Ahora dominas las tools básicas.")
    print("🚀 Próximo módulo: Tools Personalizadas Avanzadas")

if __name__ == "__main__":
    main()


