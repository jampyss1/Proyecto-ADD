# 🚀 SpaceX Data Explorer

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Aplicación web interactiva diseñada para consumir, analizar y visualizar datos históricos de lanzamientos espaciales de **SpaceX**. Este proyecto cubre el flujo completo de trabajo de un análisis de ciencia de datos, empleando **Programación Orientada a Objetos (POO)** como paradigma de desarrollo.

##  Objetivo del Proyecto

Diseñar e implementar una aplicación en Python que modele las principales etapas de un pipeline de ciencia de datos:
1. **Carga** y gestión de datos.
2. **Análisis Exploratorio (EDA)**.
3. **Preprocesamiento**.
4. **Modelado** (Machine Learning/Estadística).
5. **Evaluación** de resultados.
6. **Visualización** interactiva.

##  Stack

| Categoría | Herramienta | Justificación |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.11+ | Ecosistema de ciencia de datos |
| **Interfaz** | Streamlit | Dashboard web profesional 100% Python |
| **Visualización** | Plotly | Gráficas interactivas (zoom, hover, filtros) |
| **Fuente de Datos** | SpaceX API v4 | Datos reales, gratuitos y sin autenticación |
| **ML/Estadística** | scikit-learn, pandas | Análisis, preprocesamiento y modelado |
| **Documentación** | Sphinx + autodoc | Estándar de la industria (NumPy/SciPy) |
| **Control de Versiones** | GitHub | Historial, ramas y trazabilidad |

##  Arquitectura del Sistema

El sistema sigue una **arquitectura de 4 capas** para garantizar bajo acoplamiento y alta cohesión:

*   **Capa 4 (Presentación):** `app.py` - Dashboard Streamlit que coordina la interacción con el usuario.
*   **Capa 3 (Lógica):** Módulos de análisis, preprocesamiento, modelado, evaluación y visualización.
*   **Capa 2 (Dominio):** Clases `Lanzamiento` y `Cohete` que representan las entidades del negocio.
*   **Capa 1 (Datos):** `CargadorDatos` que consume la API externa.

### Clases Principales (POO)

El sistema está compuesto por 9 clases organizadas bajo el principio de responsabilidad única (SRP):

| Clase | Responsabilidad |
| :--- | :--- |
| `Lanzamiento` | Representa un lanzamiento individual con sus atributos. |
| `Cohete` | Representa un cohete con características técnicas y económicas. |
| `CargadorDatos` | Consume la API, deserializa JSON y construye objetos. |
| `AnalizadorExploratorio` | Genera estadísticas descriptivas y detecta valores nulos. |
| `Preprocesador` | Filtra, limpia, normaliza fechas y convierte tipos de datos. |
| `Modelador` | Implementa regresión lineal y detecta tendencias temporales. |
| `Evaluador` | Calcula métricas (R², RMSE) y valida predicciones. |
| `Visualizador` | Genera gráficas Plotly para el dashboard. |
| `Aplicacion` | Orquestador principal del pipeline. |

## Estructura del Repositorio


```text
spacex_data_explorer/
├── src/
│   ├── __init__.py
│   ├── modelos.py            # Clases de Dominio
│   ├── carga_datos.py        # Etapa 1: Carga
│   ├── analisis_eda.py       # Etapa 2: EDA
│   ├── preprocesamiento.py   # Etapa 3: Preprocesamiento
│   ├── modelado.py           # Etapa 4: Modelado
│   ├── evaluacion.py         # Etapa 5: Evaluación
│   ├── visualizacion.py      # Etapa 6: Visualización
│   └── app.py                # Orquestador (Streamlit)
├── tests/
├── requirements.txt
└── README.md

