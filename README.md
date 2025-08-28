# 🚀 Proyecto Didáctico: Aprende LangChain desde Cero hasta Experto

## 📚 Descripción
Este proyecto te llevará desde los conceptos básicos hasta el dominio avanzado de LangChain 0.3.20. Cada archivo está diseñado para enseñarte un aspecto específico de la herramienta.

## 🎯 Objetivos de Aprendizaje
- ✅ Fundamentos de LangChain
- ✅ LLMs y Prompts
- ✅ Chains y Agentes
- ✅ Memory y Conversaciones
- ✅ RAG (Retrieval Augmented Generation)
- ✅ Tools y Integraciones
- ✅ Aplicaciones Prácticas
- ✅ Mejores Prácticas

## 📁 Estructura del Proyecto

```
agente_lang_chainII/
├── 01_fundamentos/
│   ├── 01_introduccion_langchain.py    ✅ COMPLETADO
│   ├── 02_llms_basicos.py              ✅ COMPLETADO
│   └── 03_prompts_templates.py         ✅ COMPLETADO
├── 02_chains_agentes/
│   ├── 04_chains_simples.py            ✅ COMPLETADO
│   ├── 05_chains_secuenciales.py       🔄 EN DESARROLLO
│   └── 06_agentes_basicos.py           🔄 EN DESARROLLO
├── 03_memory_rag/
│   ├── 07_memory_conversaciones.py     ✅ COMPLETADO
│   ├── 08_rag_basico.py                🔄 EN DESARROLLO
│   └── 09_rag_avanzado.py              🔄 EN DESARROLLO
├── 04_tools_integraciones/
│   ├── 10_tools_basicos.py             ✅ COMPLETADO
│   ├── 11_tools_personalizados.py      🔄 EN DESARROLLO
│   └── 12_integraciones_externas.py    🔄 EN DESARROLLO
├── 05_aplicaciones_practicas/
│   ├── 13_chatbot_avanzado.py          🔄 EN DESARROLLO
│   ├── 14_analizador_documentos.py     🔄 EN DESARROLLO
│   └── 15_asistente_personal.py        🔄 EN DESARROLLO
├── 06_mejores_practicas/
│   ├── 16_optimizacion_rendimiento.py  🔄 EN DESARROLLO
│   ├── 17_manejo_errores.py            🔄 EN DESARROLLO
│   └── 18_deployment.py                🔄 EN DESARROLLO
├── utils/
│   ├── config.py                       ✅ COMPLETADO
│   └── helpers.py                      ✅ COMPLETADO
├── data/
│   └── sample_documents/               ✅ COMPLETADO
├── requirements.txt                    ✅ COMPLETADO
├── setup.py                           ✅ COMPLETADO
├── quick_start.py                     ✅ COMPLETADO
└── README.md                          ✅ COMPLETADO
```

## 🚀 Instalación

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Configurar variables de entorno:**
```bash
cp env_example.txt .env
# Editar .env con tus API keys:
# - OPENAI_API_KEY: Tu API key de OpenAI
# - PINECONE_API_KEY: Tu API key de Pinecone
# - PINECONE_ENVIRONMENT: Tu environment de Pinecone
```

3. **Probar la configuración:**
```bash
python quick_start.py
```

4. **Probar Pinecone específicamente:**
```bash
python test_pinecone.py
```

5. **Ejecutar los ejemplos:**
```bash
python 01_fundamentos/01_introduccion_langchain.py
```

## 📖 Cómo Usar Este Proyecto

### Orden de Aprendizaje Recomendado:
1. **Fundamentos** (01-03): ✅ Conceptos básicos - COMPLETADO
2. **Chains y Agentes** (04-06): 🔄 Construcción de flujos - EN PROGRESO
3. **Memory y RAG** (07-09): 🔄 Persistencia y recuperación - EN PROGRESO
4. **Tools** (10-12): 🔄 Integraciones y herramientas - EN PROGRESO
5. **Aplicaciones** (13-15): 🔄 Proyectos completos - PENDIENTE
6. **Mejores Prácticas** (16-18): 🔄 Optimización y deployment - PENDIENTE

### 📊 Progreso Actual:
- ✅ **Módulos Completados**: 6/18 (33%)
- 🔄 **En Desarrollo**: 3/18 (17%)
- ⏳ **Pendientes**: 9/18 (50%)

### Cada archivo incluye:
- 📝 Explicación detallada de conceptos
- 💻 Ejemplos de código comentados
- 🎯 Ejercicios prácticos
- ⚠️ Puntos importantes a recordar
- 🔗 Referencias adicionales

## 🎓 Niveles de Experiencia

- **Principiante**: Archivos 01-06
- **Intermedio**: Archivos 07-12
- **Avanzado**: Archivos 13-18

## 📝 Notas Importantes

- Usa LangChain 0.3.20 para compatibilidad
- **Pinecone**: Base de datos vectorial para RAG y almacenamiento de embeddings
- Cada ejemplo es independiente pero complementario
- Los ejercicios te ayudarán a consolidar el aprendizaje
- Consulta la documentación oficial para detalles específicos

## 🌲 Configuración de Pinecone

Para usar Pinecone en este proyecto:

1. **Crear cuenta en Pinecone**: https://www.pinecone.io/
2. **Obtener API Key y Environment** desde el dashboard
3. **Configurar en .env**:
   ```
   PINECONE_API_KEY=tu_api_key_aqui
   PINECONE_ENVIRONMENT=tu_environment_aqui
   PINECONE_INDEX_NAME=langchain-learning-index
   ```
4. **Probar configuración**: `python test_pinecone.py`

## 🤝 Contribuciones

Este proyecto está diseñado para aprendizaje. Si encuentras errores o tienes sugerencias, ¡son bienvenidas!

---

**¡Comienza tu viaje hacia el dominio de LangChain! 🚀**
