"""
🔍 MÓDULO 8: RAG BÁSICO CON PINECONE
====================================

En este módulo aprenderás a implementar RAG (Retrieval Augmented Generation)
usando Pinecone como base de datos vectorial para almacenar y recuperar embeddings.

🎯 OBJETIVOS DE APRENDIZAJE:
- Entender el concepto de RAG
- Configurar Pinecone para almacenar embeddings
- Crear un sistema de recuperación de información
- Implementar RAG básico con LangChain
- Optimizar búsquedas vectoriales

📚 CONCEPTOS CLAVE:
- RAG (Retrieval Augmented Generation)
- Embeddings y Vector Search
- Pinecone Integration
- Document Loading
- Text Splitting
- Similarity Search

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
from utils.pinecone_config import pinecone_config, print_pinecone_status

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document

def introduccion_rag():
    """
    🎓 Introducción al RAG
    """
    print_separator("¿QUÉ ES RAG?")
    
    print("""
    🔍 RAG (Retrieval Augmented Generation) es una técnica que combina:
    
    1. 📚 Retrieval (Recuperación): Buscar información relevante en una base de datos
    2. 🤖 Generation (Generación): Usar esa información para generar respuestas
    
    🏗️ ARQUITECTURA RAG:
    
    [Query] → [Embedding] → [Vector Search] → [Retrieved Docs] → [LLM] → [Response]
    
    📋 COMPONENTES PRINCIPALES:
    
    1. 🔤 Text Splitter: Divide documentos en chunks
    2. 🧮 Embeddings: Convierte texto en vectores
    3. 🗄️ Vector Store: Almacena y busca vectores
    4. 🔍 Retriever: Recupera documentos relevantes
    5. 🤖 LLM: Genera respuestas basadas en contexto
    
    💡 VENTAJAS:
    - Respuestas más precisas y actualizadas
    - Acceso a información específica
    - Reducción de alucinaciones
    - Escalabilidad con grandes volúmenes de datos
    """)

def configuracion_pinecone():
    """
    🌲 Configuración de Pinecone para RAG
    """
    print_step(1, "Configuración de Pinecone", "Preparando la base de datos vectorial")
    
    # Mostrar estado de Pinecone
    print_pinecone_status()
    
    # Validar configuración
    if not pinecone_config.validate_config():
        print("❌ Configuración de Pinecone incompleta")
        print("   Asegúrate de configurar PINECONE_API_KEY y PINECONE_ENVIRONMENT en .env")
        return None
    
    # Obtener o crear índice
    index = pinecone_config.get_or_create_index()
    if not index:
        print("❌ No se pudo obtener el índice de Pinecone")
        return None
    
    print("✅ Pinecone configurado correctamente para RAG")
    return index

def crear_embeddings():
    """
    🧮 Creando embeddings con OpenAI
    """
    print_step(2, "Creando Embeddings", "Configurando el modelo de embeddings")
    
    try:
        # Crear modelo de embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.openai_api_key
        )
        
        print_result("Embeddings Model", embeddings)
        
        # Probar embeddings
        test_texts = [
            "LangChain es una biblioteca para LLMs",
            "Pinecone es una base de datos vectorial",
            "RAG combina recuperación y generación"
        ]
        
        print("\n🧪 Probando embeddings:")
        for text in test_texts:
            embedding = embeddings.embed_query(text)
            print(f"📝 '{text[:30]}...' → Vector de {len(embedding)} dimensiones")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Error al crear embeddings: {e}")
        return None

def cargar_y_dividir_documentos():
    """
    📄 Cargando y dividiendo documentos
    """
    print_step(3, "Cargar y Dividir Documentos", "Preparando documentos para RAG")
    
    try:
        # Cargar documento de ejemplo
        loader = TextLoader("data/sample_documents/sample_text.txt")
        documents = loader.load()
        
        print(f"✅ Documento cargado: {len(documents)} documento(s)")
        print(f"📏 Longitud del texto: {len(documents[0].page_content)} caracteres")
        
        # Dividir documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Documentos divididos en {len(chunks)} chunks")
        print(f"📏 Tamaño promedio de chunk: {len(chunks[0].page_content)} caracteres")
        
        # Mostrar algunos chunks
        print("\n📄 Ejemplos de chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"   {chunk.page_content[:100]}...")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error al cargar documentos: {e}")
        return None

def crear_vectorstore():
    """
    🗄️ Creando vector store con Pinecone
    """
    print_step(4, "Crear Vector Store", "Almacenando embeddings en Pinecone")
    
    try:
        # Obtener índice de Pinecone
        index = pinecone_config.get_or_create_index()
        if not index:
            return None
        
        # Crear embeddings
        embeddings = OpenAIEmbeddings()
        
        # Cargar y dividir documentos
        chunks = cargar_y_dividir_documentos()
        if not chunks:
            return None
        
        # Crear vector store usando la nueva API de Pinecone
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_config.api_key)
        
        # Crear vector store
        vectorstore = LangChainPinecone.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=pinecone_config.index_name
        )
        
        print_result("Vector Store Creado", vectorstore)
        print(f"✅ {len(chunks)} documentos almacenados en Pinecone")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error al crear vector store: {e}")
        return None

def búsqueda_similitud():
    """
    🔍 Búsqueda por similitud
    """
    print_step(5, "Búsqueda por Similitud", "Probando recuperación de documentos")
    
    try:
        # Obtener vector store
        embeddings = OpenAIEmbeddings()
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_config.api_key)
        
        vectorstore = LangChainPinecone.from_existing_index(
            index_name=pinecone_config.index_name,
            embedding=embeddings
        )
        
        # Consultas de prueba
        queries = [
            "¿Qué es LangChain?",
            "¿Cuáles son las características principales?",
            "¿Qué es RAG?",
            "¿Cómo funciona la memoria en LangChain?"
        ]
        
        print("\n🔍 Probando búsquedas:")
        for query in queries:
            print(f"\n📝 Consulta: {query}")
            
            # Buscar documentos similares
            docs = vectorstore.similarity_search(query, k=2)
            
            for i, doc in enumerate(docs):
                print(f"   Resultado {i+1}: {doc.page_content[:100]}...")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return None

def rag_basico():
    """
    🤖 RAG básico con RetrievalQA
    """
    print_step(6, "RAG Básico", "Implementando RAG completo")
    
    try:
        # Crear LLM
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.7
        )
        
        # Obtener vector store
        embeddings = OpenAIEmbeddings()
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_config.api_key)
        
        vectorstore = LangChainPinecone.from_existing_index(
            index_name=pinecone_config.index_name,
            embedding=embeddings
        )
        
        # Crear retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Crear chain de RAG
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        print_result("RAG Chain", qa_chain)
        
        # Probar RAG
        questions = [
            "¿Qué es LangChain y para qué se usa?",
            "¿Cuáles son las principales características de LangChain?",
            "¿Qué es RAG y cómo funciona?",
            "¿Cuáles son los casos de uso más comunes?"
        ]
        
        print("\n🤖 Probando RAG:")
        for question in questions:
            print(f"\n❓ Pregunta: {question}")
            
            try:
                result = qa_chain({"query": question})
                print(f"🤖 Respuesta: {result['result']}")
                
                # Mostrar fuentes
                print("📚 Fuentes:")
                for i, doc in enumerate(result['source_documents'][:2]):
                    print(f"   {i+1}. {doc.page_content[:100]}...")
                    
            except Exception as e:
                print(f"❌ Error al procesar pregunta: {e}")
        
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error en RAG: {e}")
        return None

def optimizacion_rag():
    """
    ⚡ Optimización de RAG
    """
    print_step(7, "Optimización de RAG", "Mejorando el rendimiento")
    
    print("""
    ⚡ ESTRATEGIAS DE OPTIMIZACIÓN:
    
    1. 📏 Chunk Size: Ajustar tamaño de chunks según el contenido
    2. 🔄 Chunk Overlap: Configurar solapamiento para contexto
    3. 🎯 Top-K: Seleccionar número óptimo de documentos
    4. 🔍 Search Type: Elegir entre similarity, mmr, etc.
    5. 📊 Metadata: Usar metadatos para filtrado
    6. 🧮 Embedding Model: Seleccionar modelo apropiado
    """)
    
    try:
        # Configuración optimizada
        embeddings = OpenAIEmbeddings()
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_config.api_key)
        
        vectorstore = LangChainPinecone.from_existing_index(
            index_name=pinecone_config.index_name,
            embedding=embeddings
        )
        
        # Retriever optimizado
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
        
        # LLM optimizado
        llm = ChatOpenAI(
            model=config.default_model,
            temperature=0.3,  # Menos creativo, más preciso
            max_tokens=500
        )
        
        # Chain optimizada
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",  # Para documentos largos
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        
        print_result("RAG Optimizado", qa_chain)
        
        # Probar optimización
        question = "Explica detalladamente qué es LangChain y sus principales características"
        print(f"\n❓ Pregunta compleja: {question}")
        
        result = qa_chain({"query": question})
        print(f"🤖 Respuesta optimizada: {result['result']}")
        
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        return None

def ejercicios_practicos():
    """
    🎯 Ejercicios prácticos
    """
    print_step(8, "Ejercicios Prácticos", "Pon en práctica lo aprendido")
    
    ejercicios = [
        {
            "titulo": "RAG con Documentos Personalizados",
            "descripcion": "Crea un RAG con tus propios documentos",
            "objetivo": "Practicar carga y procesamiento de documentos"
        },
        {
            "titulo": "Optimización de Chunks",
            "descripcion": "Experimenta con diferentes tamaños de chunks",
            "objetivo": "Aprender optimización de text splitting"
        },
        {
            "titulo": "Búsqueda Avanzada",
            "descripcion": "Implementa diferentes tipos de búsqueda",
            "objetivo": "Practicar configuración de retrievers"
        },
        {
            "titulo": "RAG con Metadatos",
            "descripcion": "Usa metadatos para filtrado y organización",
            "objetivo": "Aprender gestión de metadatos"
        },
        {
            "titulo": "Evaluación de RAG",
            "descripcion": "Crea métricas para evaluar calidad de respuestas",
            "objetivo": "Aprender evaluación de sistemas RAG"
        }
    ]
    
    print("\n🎯 EJERCICIOS PRÁCTICOS:")
    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{i}. {ejercicio['titulo']}")
        print(f"   Descripción: {ejercicio['descripcion']}")
        print(f"   Objetivo: {ejercicio['objetivo']}")

def mejores_practicas():
    """
    ⭐ Mejores prácticas para RAG
    """
    print_step(9, "Mejores Prácticas", "Consejos para implementar RAG efectivamente")
    
    practicas = [
        "📏 Ajusta el tamaño de chunks según el tipo de contenido",
        "🔄 Usa overlap apropiado para mantener contexto",
        "🎯 Selecciona el número correcto de documentos (k)",
        "🔍 Experimenta con diferentes tipos de búsqueda",
        "📊 Usa metadatos para organizar y filtrar documentos",
        "🧮 Elige el modelo de embeddings apropiado",
        "⚡ Optimiza la configuración del LLM para RAG",
        "📝 Documenta la estructura de tus documentos",
        "🧪 Prueba con diferentes consultas y casos de uso",
        "📈 Monitorea la calidad de las respuestas"
    ]
    
    print("\n⭐ MEJORES PRÁCTICAS:")
    for practica in practicas:
        print(f"   {practica}")

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 8: RAG BÁSICO CON PINECONE")
    
    # Verificar configuración
    if not config.validate_config():
        return
    
    # Verificar Pinecone
    if not pinecone_config.validate_config():
        print("⚠️ Pinecone no está configurado. Algunas funciones no estarán disponibles.")
    
    # Contenido del módulo
    introduccion_rag()
    
    # Configuración
    configuracion_pinecone()
    embeddings = crear_embeddings()
    
    if embeddings:
        # Procesamiento de documentos
        cargar_y_dividir_documentos()
        vectorstore = crear_vectorstore()
        
        if vectorstore:
            # RAG
            búsqueda_similitud()
            rag_basico()
            optimizacion_rag()
    
    # Consolidación
    mejores_practicas()
    ejercicios_practicos()
    
    print("\n🎉 ¡Módulo 8 completado! Ahora dominas RAG básico con Pinecone.")
    print("🚀 Próximo módulo: RAG Avanzado")

if __name__ == "__main__":
    main()
