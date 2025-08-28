#!/usr/bin/env python3
"""
🚀 QUICK START - LANGCHAIN LEARNING PROJECT
==========================================

Este archivo te permite probar rápidamente tu configuración de LangChain
y ver que todo funciona correctamente antes de comenzar con los módulos.

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Prueba las importaciones básicas"""
    print("🔍 Probando importaciones básicas...")
    
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__} importado correctamente")
        
        import langchain_openai
        print("✅ LangChain OpenAI importado correctamente")
        
        import dotenv
        print("✅ Python-dotenv importado correctamente")
        
        import pinecone
        print("✅ Pinecone importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def test_configuration():
    """Prueba la configuración del proyecto"""
    print("\n⚙️ Probando configuración...")
    
    try:
        from utils.config import config, print_config_status
        print_config_status()
        
        if config.openai_api_key:
            print("✅ API Key de OpenAI configurada")
        else:
            print("⚠️  API Key de OpenAI no configurada")
            print("   Crea un archivo .env con tu API key de OpenAI")
        
        # Verificar Pinecone
        from utils.pinecone_config import pinecone_config, print_pinecone_status
        print_pinecone_status()
        
        if pinecone_config.api_key and pinecone_config.environment:
            print("✅ Configuración de Pinecone completa")
            return True
        else:
            print("⚠️  Configuración de Pinecone incompleta")
            print("   Configura PINECONE_API_KEY y PINECONE_ENVIRONMENT en .env")
            return False
            
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_basic_llm():
    """Prueba un LLM básico"""
    print("\n🤖 Probando LLM básico...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        # Crear LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        # Mensaje de prueba
        mensaje = HumanMessage(content="¡Hola! ¿Puedes confirmar que LangChain está funcionando correctamente?")
        
        print("📤 Enviando mensaje de prueba...")
        respuesta = llm.invoke([mensaje])
        
        print(f"📥 Respuesta: {respuesta.content}")
        print("✅ LLM funcionando correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en LLM: {e}")
        return False

def test_chain():
    """Prueba una chain básica"""
    print("\n🔗 Probando chain básica...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Crear LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        # Crear prompt template
        prompt = PromptTemplate(
            input_variables=["concepto"],
            template="Explica qué es {concepto} en una frase simple."
        )
        
        # Crear chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Probar chain
        resultado = chain.run("programación")
        
        print(f"📝 Resultado: {resultado}")
        print("✅ Chain funcionando correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en chain: {e}")
        return False

def show_next_steps():
    """Muestra los siguientes pasos"""
    print("\n" + "="*60)
    print("🎉 ¡CONFIGURACIÓN EXITOSA!")
    print("="*60)
    
    print("""
    ✅ Tu entorno de LangChain está listo para usar.
    
    📚 PRÓXIMOS PASOS RECOMENDADOS:
    
    1. 🚀 Comienza con el Módulo 1:
       python 01_fundamentos/01_introduccion_langchain.py
    
    2. 📖 Lee el README.md para entender la estructura del proyecto
    
    3. 🎯 Sigue el orden de los módulos:
       - Módulo 1: Introducción a LangChain
       - Módulo 2: LLMs Básicos
       - Módulo 3: Prompts y Templates Avanzados
       - Y así sucesivamente...
    
    4. 💻 Experimenta con los ejemplos y ejercicios
    
    💡 CONSEJOS:
    - Ejecuta los módulos en orden
    - Completa los ejercicios prácticos
    - Experimenta modificando los ejemplos
    - Toma notas de lo que aprendes
    
    🆘 Si tienes problemas:
    - Verifica tu API key de OpenAI
    - Asegúrate de tener todas las dependencias instaladas
    - Revisa la documentación oficial de LangChain
    """)

def main():
    """Función principal"""
    print("🚀 QUICK START - LANGCHAIN LEARNING PROJECT")
    print("=" * 50)
    
    # Pruebas básicas
    if not test_basic_imports():
        print("\n❌ Error en importaciones básicas")
        return
    
    if not test_configuration():
        print("\n❌ Error en configuración")
        return
    
    # Pruebas de funcionalidad
    if not test_basic_llm():
        print("\n❌ Error en LLM básico")
        return
    
    if not test_chain():
        print("\n❌ Error en chain básica")
        return
    
    # Todo funcionando
    show_next_steps()

if __name__ == "__main__":
    main()
