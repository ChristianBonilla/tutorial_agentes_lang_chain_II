#!/usr/bin/env python3
"""
============================================================
    MÓDULO 9: RAG AVANZADO CON TÉCNICAS MODERNAS
============================================================

🎯 OBJETIVOS:
- Implementar búsqueda híbrida (dense + sparse)
- Query expansion y reformulación
- Reranking de resultados
- RAG con múltiples fuentes
- Optimización de prompts para RAG

📚 CONTENIDO:
1. Búsqueda híbrida
2. Query expansion
3. Reranking avanzado
4. RAG multi-fuente
5. Prompts optimizados
6. Evaluación de RAG
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones de LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

# Importaciones locales
import sys
sys.path.append('..')
from utils.config import LangChainConfig
from utils.pinecone_config import PineconeConfig
from utils.helpers import print_separator, print_step, print_result

# Configuración
config = LangChainConfig()
pinecone_config = PineconeConfig()

@dataclass
class RAGResult:
    """Resultado de RAG"""
    query: str
    answer: str
    sources: List[Document]
    confidence: float
    execution_time: float

class AdvancedRAG:
    """Sistema RAG avanzado con múltiples técnicas"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=config.default_model, temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.setup_vectorstore()
    
    def setup_vectorstore(self):
        """Configurar vector store"""
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_config.api_key)
            
            self.vectorstore = LangChainPinecone.from_existing_index(
                index_name=pinecone_config.index_name,
                embedding=self.embeddings
            )
            print("✅ Vector store configurado")
        except Exception as e:
            print(f"❌ Error configurando vector store: {e}")
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Búsqueda híbrida combinando dense y sparse retrieval"""
        print_step(1, "Búsqueda Híbrida", "Implementando búsqueda combinada")
        
        # Búsqueda por similitud (dense)
        dense_results = self.vectorstore.similarity_search(query, k=k)
        
        # Búsqueda por similitud con score (dense con scores)
        dense_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Búsqueda por similitud máxima (sparse-like)
        max_marginal_results = self.vectorstore.max_marginal_relevance_search(query, k=k)
        
        # Combinar resultados
        all_results = []
        seen_ids = set()
        
        # Agregar resultados densos
        for doc in dense_results:
            if doc.page_content not in seen_ids:
                all_results.append(doc)
                seen_ids.add(doc.page_content)
        
        # Agregar resultados MMR
        for doc in max_marginal_results:
            if doc.page_content not in seen_ids:
                all_results.append(doc)
                seen_ids.add(doc.page_content)
        
        print(f"🔍 Búsqueda híbrida completada: {len(all_results)} documentos")
        return all_results[:k]
    
    def query_expansion(self, query: str) -> List[str]:
        """Expansión de consultas para mejorar recuperación"""
        print_step(2, "Query Expansion", "Generando variaciones de consulta")
        
        expansion_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Genera 3 variaciones de la siguiente consulta para mejorar la búsqueda.
            Las variaciones deben ser semánticamente similares pero usar diferentes palabras.
            
            Consulta original: {query}
            
            Variaciones:
            1. 
            2. 
            3. 
            """
        )
        
        expansion_chain = expansion_prompt.format(query=query)
        response = self.llm.invoke([HumanMessage(content=expansion_chain)])
        
        # Parsear variaciones
        lines = response.content.strip().split('\n')
        variations = []
        for line in lines:
            if line.strip() and not line.startswith('Variaciones:'):
                variation = line.strip().split('. ', 1)[-1] if '. ' in line else line.strip()
                variations.append(variation)
        
        print(f"🔄 Variaciones generadas: {variations}")
        return variations
    
    def multi_query_retrieval(self, query: str, k: int = 5) -> List[Document]:
        """Recuperación con múltiples consultas"""
        print_step(3, "Multi-Query Retrieval", "Usando múltiples consultas")
        
        # Crear retriever multi-query
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(),
            llm=self.llm
        )
        
        # Ejecutar búsqueda
        results = multi_query_retriever.get_relevant_documents(query)
        
        print(f"🔍 Multi-query retrieval: {len(results)} documentos")
        return results[:k]
    
    def contextual_compression(self, query: str, k: int = 5) -> List[Document]:
        """Compresión contextual de documentos"""
        print_step(4, "Compresión Contextual", "Comprimiendo documentos relevantes")
        
        # Crear compressor
        compressor_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Basándote en la pregunta, extrae solo la información relevante del contexto.
            
            Pregunta: {question}
            Contexto: {context}
            
            Información relevante:
            """
        )
        
        compressor = LLMChainExtractor.from_llm(self.llm, prompt=compressor_prompt)
        
        # Crear retriever con compresión
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vectorstore.as_retriever()
        )
        
        # Ejecutar búsqueda
        results = compression_retriever.get_relevant_documents(query)
        
        print(f"🗜️ Compresión contextual: {len(results)} documentos")
        return results[:k]
    
    def advanced_rag_chain(self, query: str) -> RAGResult:
        """Chain RAG avanzada con múltiples técnicas"""
        print_step(5, "RAG Avanzado", "Implementando RAG con técnicas avanzadas")
        
        start_time = time.time()
        
        # 1. Query expansion
        query_variations = self.query_expansion(query)
        
        # 2. Búsqueda híbrida con query original
        hybrid_results = self.hybrid_search(query, k=3)
        
        # 3. Multi-query retrieval
        multi_query_results = self.multi_query_retrieval(query, k=3)
        
        # 4. Compresión contextual
        compressed_results = self.contextual_compression(query, k=3)
        
        # Combinar todos los resultados
        all_docs = hybrid_results + multi_query_results + compressed_results
        
        # Eliminar duplicados
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # Prompt optimizado para RAG
        rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Eres un asistente experto. Responde la pregunta basándote únicamente en el contexto proporcionado.
            
            Si la información no está en el contexto, di "No tengo suficiente información para responder".
            
            Contexto:
            {context}
            
            Pregunta: {question}
            
            Respuesta:
            """
        )
        
        # Crear contexto
        context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])
        
        # Generar respuesta
        prompt = rag_prompt.format(context=context, question=query)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        execution_time = time.time() - start_time
        
        # Calcular confianza (simplificado)
        confidence = min(0.9, len(unique_docs) / 10)
        
        return RAGResult(
            query=query,
            answer=response.content,
            sources=unique_docs[:3],
            confidence=confidence,
            execution_time=execution_time
        )
    
    def evaluate_rag(self, test_queries: List[str]) -> Dict[str, Any]:
        """Evaluación del sistema RAG"""
        print_step(6, "Evaluación RAG", "Evaluando rendimiento del sistema")
        
        results = []
        
        for query in test_queries:
            try:
                result = self.advanced_rag_chain(query)
                results.append(result)
                print(f"✅ Query: {query[:50]}... | Tiempo: {result.execution_time:.2f}s")
            except Exception as e:
                print(f"❌ Error en query '{query}': {e}")
        
        # Métricas de evaluación
        avg_time = sum(r.execution_time for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_sources = sum(len(r.sources) for r in results) / len(results)
        
        evaluation = {
            'total_queries': len(test_queries),
            'successful_queries': len(results),
            'avg_execution_time': avg_time,
            'avg_confidence': avg_confidence,
            'avg_sources_per_query': avg_sources,
            'success_rate': len(results) / len(test_queries) * 100
        }
        
        print_result("Evaluación RAG", evaluation)
        return evaluation

def main():
    """Función principal"""
    print_separator()
    print("             MÓDULO 9: RAG AVANZADO CON TÉCNICAS MODERNAS             ")
    print_separator()
    
    # Validar configuración
    if not config.validate_config():
        print("❌ Error en configuración")
        return
    
    print("✅ Configuración validada correctamente")
    
    # Crear sistema RAG avanzado
    advanced_rag = AdvancedRAG()
    
    # Queries de prueba
    test_queries = [
        "¿Qué es LangChain y para qué se usa?",
        "¿Cuáles son las principales características de LangChain?",
        "¿Qué es RAG y cómo funciona?",
        "¿Cuáles son los casos de uso más comunes de LangChain?"
    ]
    
    # Probar RAG avanzado
    for query in test_queries:
        print(f"\n🔍 Procesando: {query}")
        result = advanced_rag.advanced_rag_chain(query)
        print(f"🤖 Respuesta: {result.answer[:100]}...")
        print(f"📚 Fuentes: {len(result.sources)} documentos")
        print(f"⏱️ Tiempo: {result.execution_time:.2f}s")
        print(f"🎯 Confianza: {result.confidence:.2f}")
    
    # Evaluar sistema
    advanced_rag.evaluate_rag(test_queries)
    
    print_separator()
    print("🎉 ¡Módulo 9 completado! Ahora dominas RAG avanzado.")
    print("🚀 Próximo módulo: Tools Personalizados")

if __name__ == "__main__":
    main()

