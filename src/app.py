"""
Módulo principal — Dashboard Streamlit (Orquestador).

Coordina todo el pipeline de ciencia de datos:
Carga → EDA → Preprocesamiento → Modelado → Evaluación → Visualización.

Clases:
    Aplicacion: Orquesta el pipeline completo usando Streamlit.
"""

import sys
from pathlib import Path

# Asegurar que el directorio padre de src/ esté en sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import streamlit as st

from src.carga_datos import CargadorDatos
from src.analisis_eda import AnalizadorExploratorio
from src.preprocesamiento import Preprocesador
from src.modelado import Modelador
from src.evaluacion import Evaluador
from src.visualizacion import Visualizador


class Aplicacion:
    """
    Orquesta todo el pipeline de ciencia de datos usando Streamlit.

    Attributes (alineados con UML)
    ----------
    cargador : CargadorDatos
    analizador_eda : AnalizadorExploratorio
    preprocesador : Preprocesador
    modelador : Modelador
    evaluador : Evaluador
    visualizador : Visualizador
    """

    def __init__(self):
        self._cargador: CargadorDatos | None = None
        self._analizador_eda: AnalizadorExploratorio | None = None
        self._preprocesador: Preprocesador | None = None
        self._modelador: Modelador | None = None
        self._evaluador: Evaluador | None = None
        self._visualizador: Visualizador | None = None

        # Internos de la vista
        self._df_crudo: pd.DataFrame | None = None
        self._df_filtrado: pd.DataFrame | None = None
        self._tipos_columna: dict | None = None

    # Etapa 0 — Selección de fuente de datos

    def _renderizar_selector_fuente(self) -> pd.DataFrame | None:
        """
        Renderiza el selector de fuente de datos en la interfaz.

        Ofrece dos opciones: subir un archivo local (CSV/JSON) o
        ingresar una URL de API REST. Retorna el DataFrame cargado
        o None si el usuario aún no ha proporcionado datos.

        Returns
        -------
        pd.DataFrame or None
        """
        st.header("Fuente de datos")
        fuente = st.radio(
            "Selecciona cómo cargar los datos:",
            ["Subir archivo", "URL de API"],
            horizontal=True,
        )

        cargador = CargadorDatos()

        if fuente == "Subir archivo":
            archivo = st.file_uploader(
                "Sube un archivo CSV o JSON",
                type=["csv", "json"],
            )
            if archivo is not None:
                try:
                    return cargador.cargar_desde_archivo(archivo)
                except Exception as e:
                    st.error(f"Error al leer el archivo: {e}")
            return None

        else:  # URL de API
            url = st.text_input(
                "URL del endpoint (debe retornar JSON):",
                placeholder="https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",
            )
            if st.button("Cargar datos") and url.strip():
                try:
                    with st.spinner("Cargando datos desde la URL…"):
                        return cargador.cargar_desde_url(url.strip())
                except Exception as e:
                    st.error(f"Error al cargar desde la URL: {e}")
            return None



    def _inicializar_componentes(self, df: pd.DataFrame) -> None:
        """
        Crea e inicializa todos los componentes del pipeline con el DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Datos cargados desde la fuente seleccionada.
        """
        self._df_crudo = df
        self._df_filtrado = df.copy()

        self._cargador = CargadorDatos()
        self._preprocesador = Preprocesador(df)
        self._tipos_columna = self._preprocesador.detectar_tipos_columna()
        self._analizador_eda = AnalizadorExploratorio(df)
        self._modelador = Modelador(df)
        self._evaluador = None  # Se crea después de entrenar el modelo
        self._visualizador = Visualizador()



    def _renderizar_barra_lateral(self) -> None:
        """
        Construye dinámicamente los filtros en el sidebar según los tipos
        de columna detectados y actualiza _df_filtrado.
        """
        with st.sidebar:
            st.header("Filtros")

            df_resultado = self._df_crudo.copy()

            # Filtros categóricos — multiselect
            for col in self._tipos_columna.get("categoricas", []):
                opciones = self._preprocesador.obtener_valores_unicos(col)
                seleccion = st.multiselect(
                    f"{col}",
                    options=opciones,
                    default=opciones,
                    key=f"cat_{col}",
                )
                if seleccion:
                    df_resultado = df_resultado[df_resultado[col].isin(seleccion)]

            # Filtros numéricos — slider de rango
            for col in self._tipos_columna.get("numericas", []):
                try:
                    minimo, maximo = self._preprocesador.obtener_rango(col)
                    if minimo == maximo:
                        continue
                    rango = st.slider(
                        f"{col}",
                        min_value=minimo,
                        max_value=maximo,
                        value=(minimo, maximo),
                        key=f"num_{col}",
                    )
                    df_resultado = df_resultado[
                        (df_resultado[col] >= rango[0]) & (df_resultado[col] <= rango[1])
                    ]
                except Exception:
                    continue

            self._df_filtrado = df_resultado

            st.caption(f"{len(self._df_filtrado):,} filas después de filtros")

    # Métricas rápidas (UML: _renderizar_metricas)

    def _renderizar_metricas(self) -> None:
        """Muestra métricas resumidas del dataset filtrado."""
        st.subheader("Metricas del dataset")
        cols = st.columns(4)
        cols[0].metric("Total filas", f"{len(self._df_filtrado):,}")
        cols[1].metric("Total columnas", len(self._df_filtrado.columns))
        cols[2].metric("Columnas numéricas", len(self._tipos_columna.get("numericas", [])))
        cols[3].metric("Columnas categóricas", len(self._tipos_columna.get("categoricas", [])))

    # Etapa 2 — EDA

    def _renderizar_etapa_eda(self) -> None:
        """Muestra el resumen exploratorio del dataset."""
        st.subheader("Analisis exploratorio (EDA)")

        analizador = AnalizadorExploratorio(self._df_filtrado)

        resumen = analizador.resumen_general()
        if resumen:
            cols = st.columns(3)
            cols[0].metric("Filas", resumen.get("total_registros", len(self._df_filtrado)))
            cols[1].metric("Columnas", resumen.get("total_columnas", len(self._df_filtrado.columns)))
            nulos = resumen.get("total_nulos", 0)
            cols[2].metric("Valores nulos", nulos)
        else:
            cols = st.columns(2)
            cols[0].metric("Filas", len(self._df_filtrado))
            cols[1].metric("Columnas", len(self._df_filtrado.columns))

        with st.expander("Estadísticas descriptivas"):
            estadisticas = analizador.estadisticas_descriptivas()
            if estadisticas is not None:
                st.dataframe(estadisticas)

        with st.expander("Valores nulos por columna"):
            nulos = analizador.detectar_nulos()
            if nulos:
                st.json(nulos)
            else:
                st.success("No se detectaron valores nulos.")

        # Detección de atípicos (solo columnas numéricas)
        numericas = self._tipos_columna.get("numericas", [])
        if numericas:
            with st.expander("Detección de valores atípicos"):
                col_atipicos = st.selectbox(
                    "Columna a analizar:",
                    numericas,
                    key="eda_atipicos",
                )
                atipicos = analizador.detectar_atipicos(col_atipicos)
                if atipicos:
                    st.warning(f"Se encontraron {len(atipicos)} valores atípicos.")
                    st.write(atipicos[:20])  # Mostrar máx 20
                else:
                    st.success("No se detectaron valores atípicos en esta columna.")

    # Gráficas (UML: _renderizar_graficas)

    def _renderizar_graficas(self) -> None:
        """Renderiza gráficas interactivas basadas en el dataset filtrado."""
        st.subheader("Visualizaciones")

        numericas = self._tipos_columna.get("numericas", [])
        categoricas = self._tipos_columna.get("categoricas", [])
        fechas = self._tipos_columna.get("fechas", [])

        # Gráfica de barras — distribución de una columna categórica
        if categoricas:
            col_cat = st.selectbox(
                "Columna categórica para gráfica de barras:",
                categoricas,
                key="viz_cat",
            )
            modelador_viz = Modelador(self._df_filtrado)
            datos_barras = modelador_viz.conteo_por_categoria(col_cat)
            fig_barras = self._visualizador.graficar_barras_comparativas(datos_barras)
            if fig_barras is not None:
                st.plotly_chart(fig_barras, use_container_width=True)

        # Comparación numérica entre categorías
        if categoricas and numericas:
            with st.expander("Comparación numérica por categoría"):
                col_grupo = st.selectbox(
                    "Agrupar por:", categoricas, key="viz_comp_cat"
                )
                col_valor = st.selectbox(
                    "Valor numérico:", numericas, key="viz_comp_num"
                )
                modelador_comp = Modelador(self._df_filtrado)
                datos_comp = modelador_comp.promedio_por_categoria(col_grupo, col_valor)
                fig_comp = self._visualizador.graficar_comparacion_numerica(datos_comp)
                if fig_comp is not None:
                    st.plotly_chart(fig_comp, use_container_width=True)

        # Gráfica de línea temporal
        if fechas and numericas:
            col_fecha = st.selectbox("Columna temporal:", fechas, key="viz_fecha")
            col_val = st.selectbox("Columna de valor:", numericas, key="viz_val")
            try:
                modelador_viz = Modelador(self._df_filtrado)
                resultado = modelador_viz.analisis_tendencia(col_fecha, col_val)
                fig_linea = self._visualizador.graficar_linea_temporal(resultado.get("serie", {}))
                if fig_linea is not None:
                    st.plotly_chart(fig_linea, use_container_width=True)
                    st.info(
                        f"Tendencia: **{resultado['tendencia']}** "
                        f"(pendiente: {resultado['pendiente']:.4f})"
                    )
            except Exception as e:
                st.warning(f"No se pudo generar la gráfica temporal: {e}")

    # Etapa de modelado

    def _renderizar_etapa_modelado(self) -> None:
        """Permite al usuario entrenar una regresión lineal sobre dos columnas."""
        st.subheader("Modelado - Regresion lineal")

        numericas = self._tipos_columna.get("numericas", [])
        if len(numericas) < 2:
            st.info("Se necesitan al menos 2 columnas numéricas para entrenar un modelo.")
            return

        col1, col2 = st.columns(2)
        col_x = col1.selectbox("Variable independiente (X):", numericas, key="model_x")
        col_y = col2.selectbox(
            "Variable dependiente (Y):",
            numericas,
            index=min(1, len(numericas) - 1),
            key="model_y",
        )

        if st.button("Entrenar modelo"):
            try:
                modelador = Modelador(self._df_filtrado)
                modelador.entrenar_regresion_lineal(col_x, col_y)
                coefs = modelador.obtener_coeficientes()
                st.success(
                    f"Modelo entrenado. Pendiente: {coefs['pendiente']:.4f} | "
                    f"Intercepto: {coefs['intercepto']:.4f}"
                )

                # --- Evaluación del modelo (UML: Evaluador) ---
                predicciones = modelador.obtener_predicciones()
                valores_reales = modelador.obtener_valores_reales()

                if predicciones is not None and valores_reales is not None:
                    self._evaluador = Evaluador(valores_reales, predicciones)
                    reporte = self._evaluador.generar_reporte()

                    st.subheader("Evaluacion del modelo")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("R²", f"{reporte['r2']:.4f}")
                    m2.metric("RMSE", f"{reporte['rmse']:.4f}")
                    m3.metric("MAE", f"{reporte['mae']:.4f}")
                    st.info(f"Diagnóstico: {reporte['diagnostico']}")

                    # Gráfica de métricas
                    datos_metricas = {
                        "R²": reporte["r2"],
                        "RMSE": reporte["rmse"],
                        "MAE": reporte["mae"],
                    }
                    fig_eval = self._visualizador.graficar_metricas_evaluacion(datos_metricas)
                    if fig_eval is not None:
                        st.plotly_chart(fig_eval, use_container_width=True)

                # --- Predicción puntual ---
                st.subheader("Prediccion")
                valor_pred = st.number_input(
                    f"Predecir {col_y} para un valor de {col_x}:",
                    value=float(self._df_filtrado[col_x].mean()),
                )
                prediccion = modelador.predecir(valor_pred)
                st.metric(f"Predicción de {col_y}", f"{prediccion:.4f}")

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {e}")

    # Punto de entrada (UML: ejecutar)

    def ejecutar(self) -> None:
        """
        Punto de entrada principal de la aplicación.

        Flujo:
        1. Configura la página Streamlit.
        2. Muestra el selector de fuente de datos.
        3. Si no hay datos: muestra instrucciones y detiene la ejecución.
        4. Inicializa todos los componentes con el DataFrame cargado.
        5. Renderiza la barra lateral de filtros.
        6. Ejecuta el pipeline completo en orden.
        """
        st.set_page_config(
            page_title="Explorador de Datos",
            layout="wide",
        )
        st.title("Explorador de Datos")
        st.markdown("Analiza cualquier dataset CSV o JSON con un pipeline de ciencia de datos.")

        df = self._renderizar_selector_fuente()
        if df is None:
            st.info("Sube un archivo CSV/JSON o ingresa una URL para comenzar el análisis.")
            st.stop()

        self._inicializar_componentes(df)

        self._renderizar_barra_lateral()

        st.divider()
        self._renderizar_metricas()
        st.divider()
        self._renderizar_etapa_eda()
        st.divider()
        self._renderizar_graficas()
        st.divider()
        self._renderizar_etapa_modelado()


if __name__ == "__main__":
    app = Aplicacion()
    app.ejecutar()
