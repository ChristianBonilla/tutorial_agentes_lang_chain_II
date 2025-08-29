"""
🌲 Configuración y Utilidades de Pinecone
========================================

Este módulo maneja la configuración y utilidades específicas para Pinecone,
la base de datos vectorial que usaremos en el proyecto.

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import os
import pinecone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class PineconeConfig:
    """
    🌲 Configuración específica para Pinecone
    """
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-learning-index")
        self.dimension = 1536  # Dimension para embeddings de OpenAI
        self.metric = "cosine"  # Métrica de similitud
        
    def validate_config(self) -> bool:
        """
        ✅ Valida la configuración de Pinecone
        
        Returns:
            bool: True si la configuración es válida
        """
        if not self.api_key:
            print("❌ PINECONE_API_KEY no está configurada")
            return False
            
        if not self.environment:
            print("❌ PINECONE_ENVIRONMENT no está configurado")
            return False
            
        print("✅ Configuración de Pinecone válida")
        return True
    
    def initialize_pinecone(self):
        """
        🚀 Inicializa Pinecone
        """
        if not self.validate_config():
            return None
            
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            print("✅ Pinecone inicializado correctamente")
            return pc
        except Exception as e:
            print(f"❌ Error al inicializar Pinecone: {e}")
            return None
    
    def get_or_create_index(self):
        """
        📊 Obtiene o crea el índice de Pinecone
        
        Returns:
            Pinecone index object
        """
        pc = self.initialize_pinecone()
        if not pc:
            return None
            
        try:
            # Verificar si el índice existe
            indexes = pc.list_indexes()
            if self.index_name in indexes.names():
                print(f"✅ Índice '{self.index_name}' encontrado")
                return pc.Index(self.index_name)
            else:
                # Crear nuevo índice
                print(f"🆕 Creando índice '{self.index_name}'...")
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
                print(f"✅ Índice '{self.index_name}' creado exitosamente")
                return pc.Index(self.index_name)
                
        except Exception as e:
            print(f"❌ Error al obtener/crear índice: {e}")
            return None
    
    def delete_index(self):
        """
        🗑️ Elimina el índice de Pinecone
        """
        try:
            pc = self.initialize_pinecone()
            if not pc:
                return
                
            indexes = pc.list_indexes()
            if self.index_name in indexes.names():
                pc.delete_index(self.index_name)
                print(f"✅ Índice '{self.index_name}' eliminado")
            else:
                print(f"⚠️ Índice '{self.index_name}' no existe")
        except Exception as e:
            print(f"❌ Error al eliminar índice: {e}")
    
    def get_index_stats(self):
        """
        📊 Obtiene estadísticas del índice
        
        Returns:
            dict: Estadísticas del índice
        """
        try:
            pc = self.initialize_pinecone()
            if not pc:
                return None
                
            index = pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"❌ Error al obtener estadísticas: {e}")
            return None

# Instancia global de configuración de Pinecone
pinecone_config = PineconeConfig()

def print_pinecone_status():
    """
    📊 Imprime el estado de Pinecone
    """
    print("\n" + "="*50)
    print("🌲 ESTADO DE PINECONE")
    print("="*50)
    print(f"API Key: {'✅ Configurada' if pinecone_config.api_key else '❌ No configurada'}")
    print(f"Environment: {pinecone_config.environment or '❌ No configurado'}")
    print(f"Index Name: {pinecone_config.index_name}")
    print(f"Dimension: {pinecone_config.dimension}")
    print(f"Metric: {pinecone_config.metric}")
    
    # Verificar conexión
    if pinecone_config.validate_config():
        try:
            pinecone.init(api_key=pinecone_config.api_key, environment=pinecone_config.environment)
            indexes = pinecone.list_indexes()
            print(f"Índices disponibles: {indexes}")
            if pinecone_config.index_name in indexes:
                print(f"✅ Índice '{pinecone_config.index_name}' existe")
            else:
                print(f"⚠️ Índice '{pinecone_config.index_name}' no existe")
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
    
    print("="*50 + "\n")

def create_sample_embeddings():
    """
    📝 Crea embeddings de ejemplo para testing
    """
    from langchain_openai import OpenAIEmbeddings
    
    try:
        # Crear embeddings
        embeddings = OpenAIEmbeddings()
        
        # Textos de ejemplo
        texts = [
            "LangChain es una biblioteca para desarrollar aplicaciones con LLMs",
            "Los LLMs son modelos de lenguaje entrenados con grandes cantidades de texto",
            "Pinecone es una base de datos vectorial para almacenar embeddings",
            "Los embeddings son representaciones vectoriales de texto"
        ]
        
        # Generar embeddings
        vectors = embeddings.embed_documents(texts)
        
        print(f"✅ {len(vectors)} embeddings creados")
        print(f"📏 Dimensión de cada embedding: {len(vectors[0])}")
        
        return texts, vectors
        
    except Exception as e:
        print(f"❌ Error al crear embeddings: {e}")
        return None, None

def test_pinecone_integration():
    """
    🧪 Prueba la integración completa con Pinecone
    """
    print("🧪 Probando integración con Pinecone...")
    
    # Obtener índice
    index = pinecone_config.get_or_create_index()
    if not index:
        print("❌ No se pudo obtener el índice")
        return False
    
    # Crear embeddings de prueba
    texts, vectors = create_sample_embeddings()
    if not vectors:
        print("❌ No se pudieron crear embeddings")
        return False
    
    try:
        # Preparar datos para Pinecone
        vectors_to_upsert = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            vectors_to_upsert.append({
                'id': f'sample_{i}',
                'values': vector,
                'metadata': {'text': text}
            })
        
        # Insertar en Pinecone
        index.upsert(vectors=vectors_to_upsert)
        print(f"✅ {len(vectors_to_upsert)} vectores insertados en Pinecone")
        
        # Probar búsqueda
        query_vector = vectors[0]  # Usar el primer embedding como query
        results = index.query(
            vector=query_vector,
            top_k=2,
            include_metadata=True
        )
        
        print("🔍 Resultados de búsqueda:")
        for match in results.matches:
            print(f"   - ID: {match.id}, Score: {match.score:.3f}")
            if 'text' in match.metadata:
                print(f"     Texto: {match.metadata['text']}")
            else:
                print(f"     Metadatos: {match.metadata}")
        
        # Limpiar datos de prueba
        index.delete(ids=[f'sample_{i}' for i in range(len(texts))])
        print("🧹 Datos de prueba eliminados")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        return False

if __name__ == "__main__":
    print_pinecone_status()
    test_pinecone_integration()
