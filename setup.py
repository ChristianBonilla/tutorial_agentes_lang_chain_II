#!/usr/bin/env python3
"""
🚀 Script de Configuración del Proyecto LangChain
================================================

Este script te ayuda a configurar todo lo necesario para comenzar
con el proyecto de aprendizaje de LangChain.

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Imprime el banner del proyecto"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 LANGCHAIN LEARNING PROJECT             ║
    ║                                                              ║
    ║              Tu viaje hacia el dominio de LangChain          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Verifica la versión de Python"""
    print("🔍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("\n📦 Instalando dependencias...")
    
    try:
        # Verificar si requirements.txt existe
        if not os.path.exists("requirements.txt"):
            print("❌ Error: No se encontró requirements.txt")
            return False
        
        # Instalar dependencias
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        return False

def setup_environment():
    """Configura el archivo de variables de entorno"""
    print("\n⚙️ Configurando variables de entorno...")
    
    env_file = ".env"
    env_example = "env_example.txt"
    
    if os.path.exists(env_file):
        print("✅ Archivo .env ya existe")
        return True
    
    if not os.path.exists(env_example):
        print("❌ Error: No se encontró env_example.txt")
        return False
    
    try:
        # Copiar archivo de ejemplo
        shutil.copy(env_example, env_file)
        print("✅ Archivo .env creado desde env_example.txt")
        print("⚠️  IMPORTANTE: Edita el archivo .env con tu API key de OpenAI")
        return True
        
    except Exception as e:
        print(f"❌ Error al crear .env: {e}")
        return False

def create_directories():
    """Crea directorios necesarios"""
    print("\n📁 Creando directorios...")
    
    directories = [
        "data/chroma_db",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def test_installation():
    """Prueba la instalación"""
    print("\n🧪 Probando instalación...")
    
    try:
        # Importar módulos principales
        import langchain
        import langchain_openai
        import dotenv
        
        print("✅ Módulos principales importados correctamente")
        
        # Probar configuración
        from utils.config import config
        print("✅ Configuración cargada correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def print_next_steps():
    """Imprime los siguientes pasos"""
    print("""
    🎉 ¡Configuración completada!
    
    📋 PRÓXIMOS PASOS:
    
    1. 🔑 Edita el archivo .env y agrega tu API key de OpenAI:
       OPENAI_API_KEY=tu_api_key_aqui
    
    2. 🚀 Ejecuta tu primer módulo:
       python 01_fundamentos/01_introduccion_langchain.py
    
    3. 📚 Lee el README.md para más información
    
    4. 🎯 Sigue el orden de los módulos para el mejor aprendizaje
    
    💡 CONSEJOS:
    - Asegúrate de tener una API key válida de OpenAI
    - Ejecuta los módulos en orden
    - Experimenta con los ejemplos
    - Completa los ejercicios prácticos
    
    🆘 Si tienes problemas:
    - Verifica tu conexión a internet
    - Asegúrate de tener Python 3.8+
    - Revisa que tu API key sea válida
    """)

def main():
    """Función principal"""
    print_banner()
    
    # Verificar Python
    if not check_python_version():
        return
    
    # Instalar dependencias
    if not install_dependencies():
        return
    
    # Configurar entorno
    if not setup_environment():
        return
    
    # Crear directorios
    create_directories()
    
    # Probar instalación
    if not test_installation():
        print("⚠️  La instalación puede tener problemas")
    
    # Próximos pasos
    print_next_steps()

if __name__ == "__main__":
    main()



