"""
📁 Configuración y Utilidades del Proyecto LangChain
===================================================

Este módulo contiene la configuración central del proyecto y utilidades comunes
que se usarán en todos los ejemplos de aprendizaje.

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Cargar variables de entorno desde .env
load_dotenv()

class LangChainConfig:
    """
    🎛️ Clase de configuración central para LangChain
    
    Esta clase maneja todas las configuraciones necesarias para el proyecto,
    incluyendo API keys, configuraciones de modelos y parámetros de desarrollo.
    """
    
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
        
        # Configuración de LangSmith (trazabilidad)
        self.langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "langchain-learning-project")
        
        # Configuración de Pinecone
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-learning-index")
        
        # Configuración de base de datos vectorial
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", "pinecone")
        self.persist_directory = os.getenv("PERSIST_DIRECTORY", "./data/vector_db")
        
        # Configuración de logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Configuración de modelos
        self.default_model = "gpt-3.5-turbo"
        self.default_temperature = 0.7
        self.default_max_tokens = 1000
        
    def validate_config(self) -> bool:
        """
        ✅ Valida que la configuración sea correcta
        
        Returns:
            bool: True si la configuración es válida, False en caso contrario
        """
        if not self.openai_api_key:
            print("⚠️  ADVERTENCIA: OPENAI_API_KEY no está configurada")
            print("   Crea un archivo .env basado en env_example.txt")
            return False
        
        print("✅ Configuración validada correctamente")
        return True
    
    def get_model_config(self, 
                        model: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> dict:
        """
        🔧 Obtiene la configuración del modelo con valores por defecto
        
        Args:
            model: Nombre del modelo a usar
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
            
        Returns:
            dict: Configuración del modelo
        """
        return {
            "model": model or self.default_model,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens
        }

# Instancia global de configuración
config = LangChainConfig()

def print_config_status():
    """
    📊 Imprime el estado actual de la configuración
    """
    print("\n" + "="*50)
    print("🔧 ESTADO DE CONFIGURACIÓN")
    print("="*50)
    print(f"OpenAI API Key: {'✅ Configurada' if config.openai_api_key else '❌ No configurada'}")
    print(f"LangSmith API Key: {'✅ Configurada' if config.langsmith_api_key else '❌ No configurada'}")
    print(f"Pinecone API Key: {'✅ Configurada' if config.pinecone_api_key else '❌ No configurada'}")
    print(f"Pinecone Environment: {config.pinecone_environment or '❌ No configurado'}")
    print(f"Pinecone Index: {config.pinecone_index_name}")
    print(f"LangChain Tracing: {'✅ Activado' if config.langchain_tracing else '❌ Desactivado'}")
    print(f"Proyecto LangSmith: {config.langchain_project}")
    print(f"Vector DB Type: {config.vector_db_type}")
    print(f"Directorio de persistencia: {config.persist_directory}")
    print(f"Nivel de logging: {config.log_level}")
    print("="*50 + "\n")

# Función de utilidad para verificar si estamos en modo de desarrollo
def is_development_mode() -> bool:
    """
    🔍 Verifica si estamos en modo de desarrollo
    
    Returns:
        bool: True si estamos en desarrollo
    """
    return os.getenv("ENVIRONMENT", "development") == "development"

# Función para obtener la ruta de datos
def get_data_path(filename: str) -> str:
    """
    📁 Obtiene la ruta completa para archivos de datos
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        str: Ruta completa del archivo
    """
    return os.path.join("data", "sample_documents", filename)

if __name__ == "__main__":
    # Ejecutar validación de configuración
    print_config_status()
    config.validate_config()
