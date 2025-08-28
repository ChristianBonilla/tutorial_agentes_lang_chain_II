"""
📝 MÓDULO 3: PROMPTS Y TEMPLATES AVANZADOS
==========================================

En este módulo aprenderás técnicas avanzadas para crear prompts efectivos
y templates reutilizables que maximicen el rendimiento de tus LLMs.

🎯 OBJETIVOS DE APRENDIZAJE:
- Crear prompts estructurados y efectivos
- Usar diferentes tipos de templates
- Implementar few-shot learning
- Optimizar prompts para diferentes tareas
- Crear prompts que generen respuestas estructuradas

📚 CONCEPTOS CLAVE:
- Prompt Engineering
- Few-shot Learning
- Output Parsers
- Prompt Templates
- Structured Outputs
- Chain of Thought

Autor: Tu Instructor de LangChain
Fecha: 2024
"""

import sys
import os
import json
from typing import List, Dict, Any

# Agregar el directorio raíz al path para importar utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import config
from utils.helpers import print_separator, print_step, print_result

# Importaciones de LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

def introduccion_prompt_engineering():
    """
    🎓 Introducción al Prompt Engineering
    """
    print_separator("PROMPT ENGINEERING")
    
    print("""
    🎯 Prompt Engineering es el arte de diseñar instrucciones efectivas para LLMs.
    
    📋 PRINCIPIOS FUNDAMENTALES:
    
    1. 🎯 Claridad: Instrucciones específicas y sin ambigüedad
    2. 📏 Concisión: Prompts que no desperdicien tokens
    3. 🏗️ Estructura: Organización lógica de la información
    4. 🎭 Contexto: Proporcionar el contexto necesario
    5. 🔄 Iteración: Mejorar prompts basándose en resultados
    
    🛠️ TÉCNICAS AVANZADAS:
    - Few-shot Learning: Ejemplos en el prompt
    - Chain of Thought: Razonamiento paso a paso
    - Role-playing: Asignar roles específicos
    - Structured Output: Respuestas en formato específico
    - Temperature Control: Ajustar creatividad vs precisión
    """)

def prompt_templates_basicos():
    """
    📝 Creando Prompt Templates básicos
    """
    print_step(1, "Prompt Templates Básicos", "Templates simples y efectivos")
    
    # Template básico
    template_basico = PromptTemplate(
        input_variables=["tema", "nivel"],
        template="Explica el concepto de {tema} a nivel {nivel}. Mantén la explicación clara y concisa."
    )
    
    print_result("Template Básico", template_basico)
    
    # Ejemplo de uso
    prompt_formateado = template_basico.format(
        tema="machine learning",
        nivel="intermedio"
    )
    
    print(f"\n📝 Prompt formateado:")
    print("-" * 50)
    print(prompt_formateado)
    print("-" * 50)
    
    return template_basico

def chat_prompt_templates():
    """
    💬 Chat Prompt Templates para conversaciones
    """
    print_step(2, "Chat Prompt Templates", "Templates para conversaciones estructuradas")
    
    # Template de chat
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto instructor de programación Python. Responde de manera clara y didáctica."),
        ("human", "Explica qué es {concepto} y da un ejemplo práctico.")
    ])
    
    print_result("Chat Template", chat_template)
    
    # Formatear el template
    mensajes = chat_template.format_messages(concepto="decoradores")
    
    print(f"\n💬 Mensajes formateados:")
    for i, mensaje in enumerate(mensajes):
        print(f"{i+1}. {type(mensaje).__name__}: {mensaje.content}")
    
    return chat_template

def few_shot_learning():
    """
    🎯 Few-shot Learning con ejemplos
    """
    print_step(3, "Few-shot Learning", "Aprendiendo con ejemplos en el prompt")
    
    # Ejemplos para few-shot learning
    ejemplos = [
        {
            "pregunta": "¿Qué es una variable?",
            "respuesta": "Una variable es un contenedor que almacena datos en memoria. Ejemplo: nombre = 'Juan'"
        },
        {
            "pregunta": "¿Qué es una función?",
            "respuesta": "Una función es un bloque de código reutilizable. Ejemplo: def saludar(): return 'Hola'"
        },
        {
            "pregunta": "¿Qué es una lista?",
            "respuesta": "Una lista es una colección ordenada de elementos. Ejemplo: numeros = [1, 2, 3]"
        }
    ]
    
    # Template para ejemplos
    ejemplo_template = PromptTemplate(
        input_variables=["pregunta", "respuesta"],
        template="Pregunta: {pregunta}\nRespuesta: {respuesta}"
    )
    
    # Template principal
    prompt_template = PromptTemplate(
        input_variables=["ejemplos", "pregunta"],
        template="""
        Eres un instructor de programación. Basándote en estos ejemplos:
        
        {ejemplos}
        
        Responde la siguiente pregunta de manera similar:
        Pregunta: {pregunta}
        Respuesta:"""
    )
    
    # Crear few-shot template
    few_shot_template = FewShotPromptTemplate(
        examples=ejemplos,
        example_prompt=ejemplo_template,
        prefix="Eres un instructor de programación. Basándote en estos ejemplos:",
        suffix="Pregunta: {pregunta}\nRespuesta:",
        input_variables=["pregunta"]
    )
    
    print_result("Few-shot Template", few_shot_template)
    
    # Ejemplo de uso
    prompt_formateado = few_shot_template.format(pregunta="¿Qué es un diccionario?")
    
    print(f"\n🎯 Prompt con ejemplos:")
    print("-" * 60)
    print(prompt_formateado)
    print("-" * 60)
    
    return few_shot_template

def output_parsers():
    """
    🔧 Output Parsers para respuestas estructuradas
    """
    print_step(4, "Output Parsers", "Generando respuestas estructuradas")
    
    # Definir esquema de respuesta
    response_schemas = [
        ResponseSchema(name="concepto", description="El concepto explicado", type="string"),
        ResponseSchema(name="definicion", description="Definición clara del concepto", type="string"),
        ResponseSchema(name="ejemplo", description="Ejemplo práctico", type="string"),
        ResponseSchema(name="nivel_dificultad", description="Nivel de dificultad (básico, intermedio, avanzado)", type="string")
    ]
    
    # Crear parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # Template con formato de salida
    template_con_parser = PromptTemplate(
        template="Explica el concepto de programación: {concepto}.\n{format_instructions}",
        input_variables=["concepto"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    
    print_result("Output Parser", output_parser)
    print(f"\n📋 Instrucciones de formato:")
    print(output_parser.get_format_instructions())
    
    return output_parser, template_con_parser

def pydantic_output_parser():
    """
    🐍 Pydantic Output Parser para respuestas tipadas
    """
    print_step(5, "Pydantic Output Parser", "Respuestas con tipos de datos definidos")
    
    # Definir modelo Pydantic
    class ConceptoProgramacion(BaseModel):
        nombre: str = Field(description="Nombre del concepto")
        definicion: str = Field(description="Definición clara y concisa")
        ejemplo: str = Field(description="Ejemplo de código práctico")
        categoria: str = Field(description="Categoría (básico, intermedio, avanzado)")
        aplicaciones: List[str] = Field(description="Lista de aplicaciones comunes")
    
    # Crear parser
    parser = PydanticOutputParser(pydantic_object=ConceptoProgramacion)
    
    # Template
    template = PromptTemplate(
        template="Explica el concepto de programación: {concepto}\n{format_instructions}",
        input_variables=["concepto"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    print_result("Pydantic Parser", parser)
    
    return parser, template

def chain_of_thought():
    """
    🧠 Chain of Thought (Razonamiento paso a paso)
    """
    print_step(6, "Chain of Thought", "Razonamiento paso a paso")
    
    cot_template = PromptTemplate(
        input_variables=["problema"],
        template="""
        Resuelve el siguiente problema paso a paso:
        
        Problema: {problema}
        
        Piensa paso a paso:
        1. Primero, analiza el problema...
        2. Luego, identifica los conceptos clave...
        3. Después, aplica la lógica...
        4. Finalmente, proporciona la solución...
        
        Solución:"""
    )
    
    print_result("Chain of Thought Template", cot_template)
    
    # Ejemplo de uso
    problema = "¿Por qué es importante el manejo de errores en programación?"
    prompt_formateado = cot_template.format(problema=problema)
    
    print(f"\n🧠 Prompt con razonamiento paso a paso:")
    print("-" * 60)
    print(prompt_formateado)
    print("-" * 60)
    
    return cot_template

def role_based_prompts():
    """
    🎭 Prompts basados en roles
    """
    print_step(7, "Role-based Prompts", "Asignando roles específicos")
    
    roles = {
        "instructor": {
            "descripcion": "Experto instructor de programación",
            "template": "Eres un instructor experto en {materia}. Explica {tema} de manera didáctica y clara, usando analogías cuando sea útil."
        },
        "code_reviewer": {
            "descripcion": "Revisor de código experto",
            "template": "Eres un revisor de código senior. Analiza el siguiente código y proporciona feedback constructivo sobre {aspecto}."
        },
        "debugger": {
            "descripcion": "Especialista en debugging",
            "template": "Eres un especialista en debugging. Analiza el error en el código y proporciona una solución paso a paso."
        },
        "architect": {
            "descripcion": "Arquitecto de software",
            "template": "Eres un arquitecto de software. Diseña una solución para {problema} considerando escalabilidad, mantenibilidad y mejores prácticas."
        }
    }
    
    print("\n🎭 ROLES DISPONIBLES:")
    for rol, info in roles.items():
        print(f"\n🔹 {rol.upper()}:")
        print(f"   Descripción: {info['descripcion']}")
        print(f"   Template: {info['template']}")
    
    return roles

def prompt_optimization():
    """
    ⚡ Optimización de prompts
    """
    print_step(8, "Optimización de Prompts", "Mejores prácticas y optimización")
    
    print("""
    ⚡ ESTRATEGIAS DE OPTIMIZACIÓN:
    
    1. 📏 Longitud:
       - Prompts más cortos = menos tokens = menor costo
       - Elimina información innecesaria
       - Usa abreviaciones cuando sea apropiado
    
    2. 🎯 Especificidad:
       - Instrucciones claras y específicas
       - Define el formato de salida esperado
       - Especifica el nivel de detalle requerido
    
    3. 🔄 Iteración:
       - Prueba diferentes versiones
       - Mide la calidad de las respuestas
       - Ajusta basándose en resultados
    
    4. 🧪 A/B Testing:
       - Compara diferentes prompts
       - Mide métricas de calidad
       - Optimiza continuamente
    
    5. 📊 Monitoreo:
       - Rastrea tokens utilizados
       - Monitorea costos
       - Evalúa calidad de respuestas
    """)

def ejercicios_practicos():
    """
    🎯 Ejercicios prácticos
    """
    print_step(9, "Ejercicios Prácticos", "Pon en práctica lo aprendido")
    
    ejercicios = [
        {
            "titulo": "Template de Análisis",
            "descripcion": "Crea un template que analice código Python y genere un reporte estructurado",
            "objetivo": "Practicar output parsers"
        },
        {
            "titulo": "Few-shot para QA",
            "descripcion": "Crea un sistema de few-shot learning para preguntas y respuestas",
            "objetivo": "Aprender few-shot learning"
        },
        {
            "titulo": "Chain of Thought",
            "descripcion": "Implementa razonamiento paso a paso para resolver problemas de lógica",
            "objetivo": "Practicar CoT"
        },
        {
            "titulo": "Role-based System",
            "descripcion": "Crea un sistema que use diferentes roles según la tarea",
            "objetivo": "Aprender role-based prompts"
        },
        {
            "titulo": "Prompt Optimizer",
            "descripcion": "Crea una función que optimice prompts automáticamente",
            "objetivo": "Aprender optimización"
        }
    ]
    
    print("\n🎯 EJERCICIOS PRÁCTICOS:")
    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{i}. {ejercicio['titulo']}")
        print(f"   Descripción: {ejercicio['descripcion']}")
        print(f"   Objetivo: {ejercicio['objetivo']}")

def mejores_practicas():
    """
    ⭐ Mejores prácticas para prompts
    """
    print_step(10, "Mejores Prácticas", "Consejos para crear prompts efectivos")
    
    practicas = [
        "🎯 Siempre especifica el rol y contexto del modelo",
        "📏 Mantén los prompts concisos pero informativos",
        "🏗️ Usa estructura clara con secciones bien definidas",
        "🔧 Especifica el formato de salida esperado",
        "🎭 Usa roles específicos para diferentes tareas",
        "🧠 Implementa Chain of Thought para problemas complejos",
        "📚 Usa few-shot learning para tareas específicas",
        "⚡ Optimiza para minimizar tokens y maximizar calidad",
        "🔄 Itera y mejora basándose en resultados",
        "📊 Monitorea el rendimiento de tus prompts"
    ]
    
    print("\n⭐ MEJORES PRÁCTICAS:")
    for practica in practicas:
        print(f"   {practica}")

def main():
    """
    🎯 Función principal del módulo
    """
    print_separator("MÓDULO 3: PROMPTS Y TEMPLATES AVANZADOS")
    
    # Verificar configuración
    if not config.validate_config():
        return
    
    # Contenido del módulo
    introduccion_prompt_engineering()
    
    # Templates básicos
    prompt_templates_basicos()
    chat_prompt_templates()
    
    # Técnicas avanzadas
    few_shot_learning()
    output_parsers()
    pydantic_output_parser()
    chain_of_thought()
    role_based_prompts()
    
    # Optimización y mejores prácticas
    prompt_optimization()
    mejores_practicas()
    ejercicios_practicos()
    
    print("\n🎉 ¡Módulo 3 completado! Ahora dominas las técnicas de prompt engineering.")
    print("🚀 Próximo módulo: Chains y Agentes")

if __name__ == "__main__":
    main()



